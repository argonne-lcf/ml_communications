from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.rank
WORLD_SIZE = comm.size
import torch
import torch.distributed as dist
import argparse
import time

def all2all(data, group, add_compute=False):
    output = torch.empty_like(data)
    dist.all_to_all_single(output, data, group=group)
    if add_compute:
        output *= 100
    return output

def print_rank0(msg):
    if RANK == 0:
        print(msg, flush=True)

## TODO: How to import this func from analytic_scaling.py?
def get_num_param_per_layer(h, h_):
    '''
        Ignored bias and norm layers. Probably insignificant. Can be added later
        ## layer_norm =  seq * h
        LLM includes: 2 * h * vocab_size (look up table and Linear head)
    '''
    QKVO = h * 4*h
    FFNN = 2 * h * h_
    return (QKVO + FFNN)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-B",    type=int, default=1,       help="")
    parser.add_argument("--seq_length", "-s",    type=int, default=4096,    help="")
    parser.add_argument("--hidden_size", "-hid", type=int, default=2048,    help="")
    parser.add_argument("--ffn_hidden_size",     type=int, default=4096,    help="")
    parser.add_argument("--num_layers", "-l",    type=int, default=24,      help="")
    parser.add_argument("--num_heads", "-hc",    type=int, default=16,      help="")
    parser.add_argument("--iter",                type=int, default=1,       help="")
    parser.add_argument("--SP",                  type=int, default=2,       help="")
    parser.add_argument("--zero",                type=int, default=0,       help="")
    parser.add_argument("--add_compute",         action="store_true",       help="add dummy compute")
    # parser.add_argument("--parallelism", "-P", type=int, default=16, help="model parallelism degree") ## Arbitrary default value
    args = parser.parse_args()

    B = args.batch_size
    s = args.seq_length
    h = args.hidden_size
    hc = args.num_heads
    hs = h // hc
    h_ = args.ffn_hidden_size
    l = args.num_layers
    SP = args.SP
    DP = WORLD_SIZE // SP
    zero = args.zero
    add_compute = args.add_compute
    dtype = torch.float16 ## {torch.bfloat16, torch.float16, etc}

    ## Global setting:
    torch.set_default_device(RANK)
    torch.set_default_dtype(dtype)

    ## init & set-up distributed group
    dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)
    sp_ranks = torch.arange(WORLD_SIZE).view(DP, SP)
    dp_ranks = sp_ranks.T
    print_rank0(f"\n")
    print_rank0(f"sp_ranks: {sp_ranks.tolist()}")
    print_rank0(f"dp_ranks: {dp_ranks.tolist()}")

    ## DP group
    for ranks in dp_ranks:
        group = dist.new_group(ranks=ranks.tolist())
        if RANK in ranks:
            dp_group = group
    ## SP group (Ulysses)
    for ranks in sp_ranks:
        group = dist.new_group(ranks=ranks.tolist())
        if RANK in ranks:
            sp_group = group

    ## Asserts
    assert WORLD_SIZE % DP == 0, "world size {WORLD_SIZE} not divisible by DP degree {DP}"
    assert WORLD_SIZE % SP == 0, "world size {WORLD_SIZE} not divisible by Ulysses degree {SP}"
    assert h % hc == 0, "hidden_dim not divisible by hc"
    assert hc % SP == 0, "head count should be divisible by ulysses-degree, unless you want to use uneven head SP(TBD)?"
    assert s % SP == 0, "sequence is not divisible by ulysses-degree, unless your want to use uneven sequence SP(TBD)?"

    ## Random Data
    torch.manual_seed(42) ## set seed for CPU and CUDA

    qkv = torch.randn(SP, s//SP, B, hc//SP, 3*hs) ## NOTE: all2all operates only on first two dimension, where the 
    ## first dim has to be participant size of all2all in order for Ulysses to work. (Q. In principal, seq and hc can be next to each other instead
    ## or will that incurr higher comm volume somehow?)
    att_out = torch.randn(SP, s//SP, B, hc//SP, hs) 
    num_param_per_layer = get_num_param_per_layer(h, h_)
    total_num_param = num_param_per_layer * l

    print_rank0(f"\n")
    print_rank0(f"num_param_per_layer (M): {num_param_per_layer // 1000**2}")
    print_rank0(f"total_num_param (M): {total_num_param // 1000**2}")

    if zero == 0:
        ## all-reduce once
        model_grad = torch.randn(total_num_param, dtype=torch.float16)
    elif zero == 1:
        ## zero1: reduce-scatter once, all-gather once
        model_grad = torch.randn(total_num_param, dtype=torch.float16)
        sharded_model_grad = torch.randn(total_num_param // WORLD_SIZE, dtype=torch.float16)
        sharded_model_weights = torch.randn(total_num_param // WORLD_SIZE, dtype=torch.float16)
        model_weights = torch.randn(total_num_param, dtype=torch.float16)
    elif zero == 2:
        ## zero2: reduce-scatter after every layer (bwd), all-gather once
        grad = torch.randn(num_param_per_layer, dtype=torch.float16)
        sharded_grad = torch.randn(num_param_per_layer // WORLD_SIZE) ## Q. what happens if the parameters are not divisible by ZERO group? 
        model_weights = torch.randn(total_num_param, dtype=torch.float16)
        sharded_model_weights = torch.randn(total_num_param // WORLD_SIZE, dtype=torch.float16)
    elif zero == 3:
        ## every layer: all-gather (fwd + bwd), reduce-scatter (bwd)
        grad = torch.randn(num_param_per_layer, dtype=torch.float16)
        sharded_grad = torch.randn(num_param_per_layer // WORLD_SIZE)
        weights = torch.randn(num_param_per_layer, dtype=torch.float16)
        sharded_weights = torch.randn(num_param_per_layer // WORLD_SIZE, dtype=torch.float16)

    print_rank0(f"\n")
    print_rank0(f"qkv: {qkv.shape}")
    print_rank0(f"att_out: {att_out.shape}")
    print_rank0(f"\n")

    ## Pure Comm Model
    ## TODO: how to make the buffer customizable? Often, the ones that happen at once has configurable buffer and async. 
    strt = time.time()
    for i in range(args.iter):
        print_rank0(f"Running {i}th iteration...")
        ## pure comm (fwd)
        for l in range(l):
            if zero == 3:
                dist.all_gather_into_tensor(weights, sharded_weights)

            all2all(qkv, group=sp_group, add_compute=add_compute) ## first
            all2all(att_out, group=sp_group, add_compute=add_compute) ## second

            if zero == 3:
                dist.barrier() ## simulate overlap of compute and comm. Technically, here we offseted communication 
                # by once, which isn't exact pattern in zero but very similar. 

        ## pure comm (bwd)
        for l in range(l):
            if zero == 3:
                dist.all_gather_into_tensor(weights, sharded_weights)

            all2all(att_out, group=sp_group, add_compute=add_compute) ## second
            all2all(qkv, group=sp_group, add_compute=add_compute) ## first

            if zero == 2:
                ## Zero2 has to scatter at every layer or similar granular level.
                dist.reduce_scatter_tensor(sharded_grad, grad, op=dist.ReduceOp.AVG, async_op=True)
            if zero in {2, 3}:
                ## zero2: if reduce_scatter cannot finish in time, your memory will grow.
                dist.barrier() ## simulate overlap of compute and comm. Technically, here we offset communication 
                # by one, which isn't exact pattern in zero but very similar. 

        ## Regarding all-gather: Q. Is adam operation usually/always fused? 
        # If so, then all-gather has to happen once at the end of the batch to gather the updated parameters. 
        # Otherwise, it can happen asyncrhnously during backpropagation. One can check the implementation of 
        # deepspeed or FSDP's zero to confirm.
        if zero == 0:
            dist.all_reduce(model_grad, op=dist.ReduceOp.AVG, group=dp_group, async_op=True)
        elif zero == 1:
            dist.reduce_scatter_tensor(sharded_model_grad, model_grad, op=dist.ReduceOp.AVG, async_op=True) ## flexible buffer size & async
            ## Hidden: update master weight with sharded_model_grad
            dist.all_gather_into_tensor(model_weights, sharded_model_weights, async_op=True)
        elif zero == 2:
            dist.all_gather_into_tensor(model_weights, sharded_model_weights, async_op=True)
        dist.barrier() ## update parameters.

    tot_time = time.time() - strt
    print_rank0(f"\n")
    print_rank0("Benchmark Finished.")
    print_rank0(f"tot_time: {tot_time}")

## TODO:
## timers (concisely). Q. How to elegantly time async operations? 
## Check multi-node func
## Q. Why is Zero123 faster than zero0? Is there something wrong with all-reduce? Try to minimally reproduce the problem. 

# timers (get-max)?
# sanity check by profiling? 