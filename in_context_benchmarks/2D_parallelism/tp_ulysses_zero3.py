import os
from mpi4py import MPI

from benchmark_utils import (
    time_and_save_to_dict, get_backend, get_time_statistics, get_flop_statistics, 
    print_rank0, log_and_print_rank0, synchronize, create_new_stream
)

import torch
import torch.distributed as dist
import torch.distributed.device_mesh
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import profile, ProfilerActivity
import numpy as np

import time
import math
import argparse
import logging
from functools import partial
import pytz
from datetime import datetime

if torch.xpu.is_available():
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch


parser = argparse.ArgumentParser(
    description="parse arguments for model and parallelism configurations")
## TODO: Add group arguments such as model related hidden_dim... etc.
parser.add_argument("-s", "--sequence_length", type=int, default=4608,
                    help="Maximum sequence length. The size of the ALLGATHER buffer")
parser.add_argument("-d", "--hidden_dimension", type=int, default=9216,
                    help='Hidden dimension for the matrix multiplication. Proxy for the' 'model size.')
parser.add_argument("-hc", "--head-count", type=int, default=32,
                    help='number of head count for multi-headed-attention.')
parser.add_argument("-it", "--iterations", type=int, default=3,
                    help="number of iterations for the timing loop")
parser.add_argument("-wit", "--warmup-iter", type=int, default=2,
                    help="number of warmup iterations")
parser.add_argument("-n_layers", "--number_of_transformer_layers", type=int, default=80,
                    help="Number of transformer layers")
parser.add_argument("--micro-batch-size", type=int, default=1,
                    help='local batch size per model instance')
parser.add_argument("-p", "--precision", type=str, default="float32",
                    help="Data type for the elements of a tensor. float32 and bfloat16 supported.", )
parser.add_argument("-dvc", "--device", type=str, default="cuda",
                    help="Device type. cuda and xpu supported.")
parser.add_argument("-f", "--log_fpth", help="Output file name",
                    type=str, default="tensor_parallel.log")
parser.add_argument("--logging", help="Switch logging on", action='store_true')
parser.add_argument('--save', action='store_true', 
                    help='Save detail results in npy format. Generates huge files, use' 'with caution') ## 
parser.add_argument("--trace", action='store_true', help='use pytorch profiler to trace')
parser.add_argument("-sp_switch", "--sequence-parallel-switch", action='store_true',
                    help="Switch sequence parallelism on or off")
parser.add_argument("-bucket", "--grad-bucket-size", type=int, default=5e8,
                    help='Buket size for gradient sync that occurs across DP and ulysses group. The maximum bucket size is the number of parameters')
parser.add_argument("-TP", "--tensor-parallel-degree", type=int, default=1,
                    help='Model weights and activations are distributed across the number of tensor parallel degree ranks')
parser.add_argument("--ulysses-degree", type=int, default=1,
                    help='Ulysses parallelism deree. input and activations are distributed across the sequence dimension. Ulysses is incompatible with TP')
parser.add_argument('--use-zero3', action='store_true',
                    help='Enable parameter sharding and asynchronously all-gather parameters for the next layers. Thus, next layer must wait until the all-gather is completed')
parser.add_argument('--include-flash-attention', action='store_true', 
                    help='time and benchmark flash attention kernel by using F.scaled_dot_product_attention')
args = parser.parse_args()

# ## debug_mode
# args.device = 'cpu'
# args.n_layers = 4
# args.sequence_length = 12
# args.hidden_dimension = 48
# args.head_count = 12
# args.number_of_transformer_layers = 4
# args.use_zero3 = True
# args.tensor_parallel_degree = 1
# args.data_parallel_degree = 12
# os.environ['USE_TORCHRUN'] = '1'

## TODO: just pass-in device mesh only or will getting the attributes cause some overhead?
def emulate_transformer_layer(
    loc_input: Tensor,
    x: Tensor,
    full_input_shape: tuple[int],
    lst_weights: tuple[Tensor],
    parallelism_degrees: tuple[int],
    comm_groups: tuple[dist.ProcessGroup],
    timed: partial,
    stream_for_zero: torch.Stream,
) -> Tensor:
    r"""Emulate single transfromer layer and return the output
    
    Args:
        local_input: [s/tp/spu, b, h] if sequence_parallel else [s, b, h]
        x: [s, b, h] <-- out-buffer for comm
    
    Note:
    """
    S, B, hc, hs = full_input_shape
    H = hc * hs
    W_qkv, W_o, W_h_4h, W_4h_h = lst_weights
    TP, SPU, DP = parallelism_degrees
    SPU_DP = SPU * DP
    tp_group, spu_group, dp_group, spu_dp_group = comm_groups
    seq_dim = 0
    hc_dim = 2
    qkv = 3
    ## FIXME: to support zero3, we will need to modify the timer to use .elapsed_time() which can only be calculated after the layer has finished running (ideally at the end of the model). To do this on aurora, it requires the latest xpu version or you can just create a different version to support overlap of zero3 on polaris. 
    if args.use_zero3:  # prefetch W_qkv
        with stream_for_zero:
            W_qkv_lst = [torch.empty_like(W_qkv) for _ in range(SPU_DP)]
            timed('zero3 all-gather W_qkv', 
                  lambda: dist.all_gather(W_qkv_lst, W_qkv, group=spu_dp_group, async_op=True)
            )
            W_qkv = torch.cat(W_qkv_lst, dim=1)
    if args.sequence_parallel_switch:
        timed('SP all-gather 1',
              lambda: dist.all_gather_into_tensor(x, loc_input, group=tp_group)
        )
    # W_qkv (GEMM) -> [s, b, 3h]
    x = timed('W_qkv', lambda: torch.matmul(x, W_qkv))

    if args.use_zero3:  # prefetch W_o
        with stream_for_zero:
            W_o_lst = [torch.empty_like(W_o) for _ in range(SPU_DP)]
            timed('zero3 all-gather W_o',
                  lambda: dist.all_gather(W_o_lst, W_o, group=spu_dp_group, async_op=True)
            )
            W_o = torch.cat(W_o_lst, dim=0)
    if SPU > 1:
        all2all_inp_lst = [y.contiguous() for y in x.tensor_split(SPU, dim=hc_dim)]
        all2all_out_lst = [torch.empty_like(x) for x in all2all_inp_lst]
        timed('ulysses pre-att all2all',
              lambda: dist.all_to_all(all2all_out_lst, all2all_inp_lst, group=spu_group)
        )
        x = torch.cat(all2all_out_lst, dim=seq_dim)
    q, k, v = x.view(S, B, hc//TP, qkv, hs).permute(1, 2, 0, 3, 4).unbind(-2)  # -> (b, hc, s, hs)

    # Flash Attention -> (s, b, h//TP)
    if args.include_flash_attention:  
        att_out = timed('flash attention', lambda: F.scaled_dot_product_attention(q, k, v))
    else:
        att_out = q
    att_out = att_out.permute(0, 2, 1, 3).reshape(S, B, H//TP)

    if SPU > 1:
        all2all_inp_lst = [y.contiguous() for y in att_out.tensor_split(SPU, dim=seq_dim)]
        all2all_out_lst = [torch.empty_like(x) for x in all2all_inp_lst]
        timed('ulysses post-att all2all',  # TODO: can we make this the kernel name in trace? 
              lambda: dist.all_to_all(all2all_out_lst, all2all_inp_lst, group=spu_group)
        )
        x = torch.cat(all2all_out_lst, dim=seq_dim)
    # W_out (GEMM) -> [s, b, h]
    x = timed('W_o', lambda: torch.matmul(att_out, W_o)) 

    if args.use_zero3:  # prefetch W_h_4h
        with stream_for_zero:
            W_h_4h_lst = [torch.empty_like(W_h_4h) for _ in range(SPU_DP)]
            timed('zero3 all-gather W_h_4h',
                  lambda: dist.all_gather(W_h_4h_lst, W_h_4h, group=spu_dp_group, async_op=True)
            )
            W_h_4h = torch.cat(W_h_4h_lst, dim=1)
    if args.sequence_parallel_switch:
        timed('SP reduce-scatter 1',
              lambda: dist.reduce_scatter_tensor(loc_input, x, group=tp_group)
        )
    elif TP > 1:
        timed('TP all-reduce 1', lambda: dist.all_reduce(x, group=tp_group))
    # Skipping dropout and Norm
    
    if args.sequence_parallel_switch:
        timed('SP all-gather 2', 
              lambda: dist.all_gather_into_tensor(x, loc_input, group=tp_group)
        )
    # MLP (Up projection) -> [s, b, 4h]
    partial_proj = timed('W_h_4h', lambda: torch.matmul(x, W_h_4h)) 
    if args.use_zero3:  # prefetch W_4h_h
        W_4h_h_lst = [torch.empty_like(W_4h_h) for _ in range(SPU_DP)]
        timed('zero3 all-gather W_4h_h',
              lambda: dist.all_gather(W_4h_h_lst, W_4h_h, group=spu_dp_group, async_op=True)
        )
        W_4h_h = torch.cat(W_4h_h_lst, dim=0)

    # MLP (Down projection) -> [s, b, h]
    x = timed('W_4h_h', lambda: torch.matmul(partial_proj, W_4h_h))
    if args.sequence_parallel_switch:
        timed('SP reduce-scatter 2',
              lambda: dist.reduce_scatter_tensor(loc_input, x, group=tp_group)
        )
    elif TP > 1:
        timed('TP all-reduce 2', lambda: dist.all_reduce(x, group=tp_group))

    return x


def main():
    ## FIXME: torchrun not viable with below method
    rank = int(MPI.COMM_WORLD.Get_rank())
    world_size = int(MPI.COMM_WORLD.Get_size())

    ## set default device and data type
    if args.device == "xpu":
        import intel_extension_for_pytorch as ipex
        import oneccl_bindings_for_pytorch
    if args.precision == "float32":
        data_type = torch.float32
        data_type_multiplier = 32 ## 32 Bits = 4 Bytes
    elif args.precision == "float16":
        data_type = torch.float16
        data_type_multiplier = 16
    elif args.precision == "bfloat16":
        data_type = torch.bfloat16
        data_type_multiplier = 16 ## 16 Bits

    torch.distributed.init_process_group(
        backend=get_backend(args.device),
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    log_and_print_rank0(f"args: {args}")
    log_and_print_rank0(f"world_size: {world_size}")

    ## Initialize logging
    if args.logging:
        ## TODO: just set log to file on all ranks and log only one rank?
        # log_fpthpath = os.path.join(args.log_directory, args.log_fpth)
        log_dir = os.path.dirname(args.log_fpth)
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            logging.basicConfig(filename=args.log_fpth, filemode="a", level="INFO")
        else:
            logging.basicConfig(level="INFO")
    central_tz = pytz.timezone('America/Chicago')
    # Get current time in Central Time
    central_time = datetime.now(central_tz)
    logging.info(f"Central Time: {central_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    dist.barrier()
    logging.info(f"rank {rank}/{world_size}")

    ## Initialize communication group
    TP = args.tensor_parallel_degree
    SPU = args.ulysses_degree
    B = args.micro_batch_size
    hc = args.head_count
    assert world_size % TP % SPU == 0, \
        'Cannot infer DP as world_size is indivisible by model parallel degree'
    DP = world_size // TP // SPU
    if TP == 1:
        logging.info('turning off sequence parallel switch due to TP=1')
        args.sequence_parallel_switch = False  # SP builds on top of TP
    if args.use_zero3 and (SPU == 1 and DP == 1):
        logging.info('turning off zero3 switch due group size of 1')
        args.use_zero3 = False  # SP builds on top of TP
    assert not (TP > 1 and SPU > 1), 'Enabling both TP and ulysses is forbidden' 
    assert (hc % TP == 0) and (hc % SPU == 0), 'head count is indivisible by TP or ulysses'
    if TP > 1:  # TP with or without sequence parallelism
        mesh_shape = [TP, DP]
        mesh_dim_names = ['TP', 'DP']
    elif SPU > 1:  # Ulysses
        mesh_shape = [SPU, DP]
        mesh_dim_names = ['SPU', 'DP']
    else:  # DP only
        mesh_shape = [DP]
        mesh_dim_names = ['DP']
    device_mesh = torch.distributed.device_mesh.init_device_mesh(
        device_type=args.device,
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names
    )
    for mesh_name in mesh_dim_names:
        comm_group = device_mesh[mesh_name].get_group()
        all_ranks = dist.get_process_group_ranks(comm_group)
        ## FIXME: is the above in group rank or global rank?
        logging.info(f'{mesh_name}: {all_ranks}')

    ## set default device adn data type
    torch.set_default_dtype(data_type)
    torch.set_default_device(rank)  
    ## Issue fixed: we need to manually send tensor to each local_rank
    ## FIXME: 
    # 1. Is rank and world_size here local or global? 
    # 2. does it differ from dist.get and comm.get?

    ## Initialize input and buffers
    S = args.sequence_length
    H = args.hidden_dimension
    qkv = 3
    stable_std = 0.01
    # partition sequence if sequence parallelism is enabled
    if args.sequence_parallel_switch or SPU > 1:
        assert S % TP % SPU == 0, \
            'sequence length must be dividable by TP or ulysses degree, whichever one is enabled'
        ## TODO: call it local_input, or loc_x or what? 
        loc_input = torch.randn(S//TP//SPU, B, H) * stable_std
    else:
        loc_input = torch.randn(S, B, H) * stable_std
    x = torch.randn([S, B, H]) * stable_std
    if args.use_zero3:
        sharded_hidden_dim = H//TP//DP//SPU
    else:
        sharded_hidden_dim = H//TP
        
    ## Initialize weights with stable std
    n_layers = args.number_of_transformer_layers
    W_qkv = torch.randn(H, qkv*sharded_hidden_dim) * stable_std
    W_o = torch.randn(sharded_hidden_dim, H) * stable_std
    W_h_4h = torch.randn(H, 4*sharded_hidden_dim) * stable_std
    W_4h_h = torch.randn(4*sharded_hidden_dim, H) * stable_std
    num_total_parameters = n_layers * (
        W_qkv.numel() + W_o.numel() + W_h_4h.numel() + W_4h_h.numel()
    )
    log_and_print_rank0(f"Parameters = {num_total_parameters / 1e9} Billions")

    ## Initialize buckets for the gradient synchronization loop
    highest_bucket_size = args.grad_bucket_size
    if highest_bucket_size > num_total_parameters:
        highest_bucket_size = num_total_parameters
        log_and_print_rank0('Bucket size is too big compared to num parameters.'
                            'Adjusting bucket size to the max num parameters')
    n_iter_grad_sync = math.ceil(num_total_parameters / highest_bucket_size)
    allreduce_grad = torch.randn(int(highest_bucket_size))
    if args.use_zero3:
        scattered_grad_buffer = torch.randn(int(highest_bucket_size) // SPU // DP)

    ## Start profiling if enabled
    N_timing_loop = args.iterations
    if args.trace:
        log_and_print_rank0("Profiling...")
        if args.device == 'cuda':
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        elif args.device == "xpu":
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]
        else:
            raise NotImplementedError()
        from torch.profiler import schedule
        # my_schedule = schedule(wait=0,
        #                        warmup=args.warmup_iter, 
        #                        active=N_timing_loop - args.warmup_iter)
        # my_schedule = schedule(wait=1,
        #                        warmup=1, 
        #                        active=1)
        #                schedule=my_schedule,
        ## TODO: fix schedule making our trace empty
        prof = profile(activities=activities, 
                       record_shapes=True)
        prof.start()

    ## Create a dictionary to log time
    hs = H // hc
    assert H % hc == 0
    assert hc % TP == 0

    ## Run and time pseudo parallel transformer
    stream_for_zero = create_new_stream()
    tp_group = device_mesh['TP'].get_group() if TP > 1 else None
    spu_group = device_mesh['SPU'].get_group() if SPU > 1 else None
    dp_group = device_mesh['DP'].get_group()

    if SPU > 1 and args.use_zero3:
        spu_dp_group = None  # HACK: global_group
    else:
        spu_dp_group = device_mesh['DP'].get_group()
    dict_ops_time = {}
    timed = partial(time_and_save_to_dict, dict_time=dict_ops_time)
    for i in range(N_timing_loop):
        ## TODO: Time each loop?
        for l in range(n_layers):
            print(f"At {i}th loop, running layer {l}")
            emulate_transformer_layer(
                loc_input=loc_input,
                x=x,
                full_input_shape=(S, B, hc, hs),
                lst_weights=(W_qkv, W_o, W_h_4h, W_4h_h),
                parallelism_degrees=(TP, SPU, DP),
                comm_groups=(tp_group, spu_group, dp_group, spu_dp_group),
                timed=timed,
                stream_for_zero=stream_for_zero,
            )
            print(f"At {i}th loop, finished layer {l}")

        synchronize()
        log_and_print_rank0(f"doing grad sync of iter {i}")
        ## Grad sync  # TODO: make grad-sync async?
        for _ in range(n_iter_grad_sync):
            if args.use_zero3:
                timed('grad reduce-scatter (1 bucket)', 
                      lambda: dist.reduce_scatter_tensor(scattered_grad_buffer, allreduce_grad, group=spu_dp_group)
                )
                timed('param update all-gather (1 bucket)', 
                      lambda: dist.all_gather_into_tensor(allreduce_grad, scattered_grad_buffer, group=spu_dp_group)
                )
            else:
                timed('grad all-reduce (1 bucket)', 
                      lambda: dist.all_reduce(allreduce_grad, group=spu_dp_group))
        
        if args.trace:
            # synchronize()
            # print(f"profiler stepping at {i}")
            prof.step()

    synchronize()
    if args.trace:
        prof.stop()
        prof.export_chrome_trace(
            f"{log_dir}/{args.trace}-{rank}-of-{world_size}.json")
    synchronize()

    tp_allreduce_data_volume = S * B * H * data_type_multiplier
    sp_allgather_data_volume = (args.sequence_length // TP * args.hidden_dimension * data_type_multiplier)

    if rank == 0:
        logging.info(f"==== Main Results ====\n")
        logging.info(f"Running with {args.precision} data type")
        logging.info(f"==== List of Arguments ====")
        logging.info(f"Sequence Length = {args.sequence_length}")
        logging.info(f"Hidden Dimension = {args.hidden_dimension}")
        logging.info(f"Number of transformer layers = {args.number_of_transformer_layers}")
        logging.info(f"Precision Type = {args.precision}")
        logging.info(f"SP switch = {args.sequence_parallel_switch}")
        logging.info(f"TP Degree = {TP}") 
        logging.info("==== List of Arguments ====")
        logging.info(f"Shape of the (Q,K,V) atten. matrix = {W_qkv.shape}")
        logging.info(f"Shape of the WO atten. matrix = {W_o.shape}")
        logging.info(f"Shape of the Weight matrix (H --> 4H)= {W_h_4h.shape}")
        logging.info(f"Shape of the Weight matrix (4H --> H)= {W_4h_h.shape}")
        logging.info(f"Parameters (per rank) = {num_total_parameters / 1e9} Billions")
        logging.info(f"N_iter_grad_sync = {n_iter_grad_sync}")

        ## TODO use variables or create some functions...
        logging.info(f"Allgather buffer size = {(args.sequence_length * args.hidden_dimension * data_type_multiplier) / 8 / 1e6} MB")
        logging.info(f"Grad Sync Allreduce bucket size = {(highest_bucket_size * data_type_multiplier) / 8 / 1e6} MB") 
        logging.info(f"TP Allreduce 1 data volume per layer per iteration = {(tp_allreduce_data_volume ) / 8 / 1e6} MB")

        logging.info("\n==== Timings per transformer layer ====")
        flop_logs = []
        # hidden_dim is sharded after col parallel GEMM but not before
        dict_gemm_inp_shapes = {
            'W_qkv': [S//SPU, B, H],
            'W_o': [S//SPU, B, H//TP],
            'W_h_4h': [S//SPU, B, H],
            'W_4h_h': [S//SPU, B, H//TP],
        }
        ## TODO: Flash Attention Flops
        for op_name, lst_time in dict_ops_time.items():
            logging.info(get_time_statistics(op_name, lst_time, args.warmup_iter))
            if 'W_' in op_name and len(op_name.split()) == 1:
                # assuming weight tensor name is also an op_name and contiguous 
                weight_shape = eval(op_name).shape
                gemm_input_shapes = (dict_gemm_inp_shapes[op_name], weight_shape)
                flop_logs.append(get_flop_statistics(op_name, gemm_input_shapes, 
                                                     lst_time, args.warmup_iter))
        logging.info("\n==== Flops per transformer layer ====")
        for flop_log in flop_logs:
            logging.info(flop_log)

    if args.save:
        result_dir = os.path.join(log_dir, "timings")
        print_rank0("saving to:", args.log_fpth)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        np.save(os.path.join(result_dir, f"rank_{rank}"), dict_ops_time)
    dist.barrier()
    exit()

if __name__ == "__main__":
    main()