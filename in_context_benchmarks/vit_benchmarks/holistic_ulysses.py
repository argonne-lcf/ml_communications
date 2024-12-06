from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.rank
WORLD_SIZE = comm.size
SP = WORLD_SIZE ## Ulysses_degree
DP = WORLD_SIZE // SP ## TODO: DP TBD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import time

SEQUENCE_SHARD_LIST = None

def get_sequence_shard_list():
    return SEQUENCE_SHARD_LIST

def set_sequence_shard_list(shard_list: list):
    global SEQUENCE_SHARD_LIST
    SEQUENCE_SHARD_LIST = shard_list

## TODO: reduce the number of permute & transpose?

def even_all2all(input, group):
    out = torch.empty_like(input, device=RANK) ## Q. How come during backgward(), the default device is not used? 
    dist.all_to_all_single(out, input, group=group)
    return out

def uneven_all2all(is_first_all2all, input, group):
    if is_first_all2all:
        _SP, local_seq, B, local_hc, hs = input.shape ## _SP avoids SP variable collision.
        input_splits = [1]*SP
        output_splits = get_sequence_shard_list()
        s = sum(output_splits)
        out = torch.empty(s, B, local_hc, hs, device=RANK)
    else:
        s, B, local_hc, hs = input.shape
        input_splits = get_sequence_shard_list()
        output_splits = [1]*SP
        local_seq = get_sequence_shard_list()[RANK]
        out = torch.empty(SP, local_seq, B, local_hc, hs, device=RANK) ## TODO: Do we need this SP dimension?
    dist.all_to_all_single(out, input, group=group, input_split_sizes=input_splits, output_split_sizes=output_splits)
    return out

class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gather_idx, scatter_idx, group):
        '''
            input: []
        '''
        ctx.group = group
        ctx.gather_idx = gather_idx
        ctx.scatter_idx = scatter_idx
        is_first_all2all = gather_idx < 2
        is_uneven_sequence = (get_sequence_shard_list() is not None)
        if is_first_all2all:
            local_s, B, hc, hs = input.shape
            local_hc = hc//SP
            input = (
                input
                .view(local_s, B, SP, local_hc, hs) ## split head by SP degree
                .permute(2, 0, 1, 3, 4) ## [sp, local_s, B, hc/sp, 3*hs]
                .contiguous()
            )
            if is_uneven_sequence:
                out = uneven_all2all(is_first_all2all, input, group)
            else:
                out = even_all2all(input, group)
        else:
            s, B, local_hc, hs = input.shape
            hc = local_hc * SP
            input = input.contiguous()
            if is_uneven_sequence:
                local_s = get_sequence_shard_list()[RANK]
                out = uneven_all2all(is_first_all2all, input, group)
            else:
                local_s = s // SP
                out = even_all2all(input, group)
            out = (out
                .permute(1, 2, 0, 3, 4) ## [local_s, B, sp, hc/sp, hs]
                .contiguous() 
                .flatten(2, 3) ## [local_s, B, hc, hs]
            )
        return out
    
    @staticmethod
    def backward(ctx, *grad_output):
        return All2All.apply(*grad_output, ctx.scatter_idx, ctx.gather_idx, ctx.group), None, None, None ## trigger the conjugate collective communication

class DistributedTransformer(nn.Module):
    def __init__(self, emb_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.get_qkvs = nn.ModuleList(nn.Linear(emb_dim, 3*emb_dim) for _ in range(num_layers)) ## Fused qkv layer w/ bias term
        self.dense_outs = nn.ModuleList(nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)) ## dense layer on att_out
        
    def forward(self, x:torch.tensor, sp_group=None, comm_only:bool=False) -> torch.tensor:
        s, B, hc, hs = x.size()
        for get_qkv, dense_out in zip(self.get_qkvs, self.dense_outs):
            x = x.view(s, B, hc*hs) ## [local_s, B, hc*hs] flatten to perform full GEMM
            qkv = get_qkv(x).view(s, B, hc, 3*hs) ## [local_s, B, 3*hc*hs] project to 3xdim for qkv
            
            if sp_group is not None:
                qkv = All2All.apply(qkv, 0, 2, sp_group) ## First all2all

            ## ATTENTION MECHANISM
            qkv = qkv.permute(1, 2, 0, 3) ## [B, hc/sp, local_s, 3*hs] Follow torch attention's dimension syntax
            q, k, v = qkv.tensor_split(3, dim=-1) ## [B, hc/sp, local_s, hs] x3 Extract qkv
            att_out = F.scaled_dot_product_attention(q, k, v) ## [B, hc/sp, s, hs]
            x = att_out.permute(2, 0, 1, 3) ## [s, B, hc/sp, hs] Follow all2all syntax

            if sp_group is not None:
                x = All2All.apply(x, 2, 0, sp_group) ## Second all2all: 

            ## final linear projection
            x = x.view(s, B, hc*hs)
            x = dense_out(x).view(s, B, hc, hs)
        return x

if __name__ == "__main__":
    ## TODO: add argparse
    # import argparse
    B = 1
    s = 4099
    hid_dim = 4096
    hc = 32
    hs = hid_dim // hc
    num_layers = 24
    unit_test = True
    dtype = torch.float16 ## {torch.bfloat16, torch.float16, etc}

    ## Global setting:
    # torch.backends.cuda.enable_flash_sdp(enabled) ## globally enable FA2. By default, let pytorch choose the most optimal one
    torch.set_default_device(RANK)
    torch.set_default_dtype(dtype)

    torch.manual_seed(42) ## seed for both CPU and CUDA
    x = torch.randn(s, B, hc, hs)
    label = torch.randn(s, B, hc, hs)
    embedding_dim = hc * hs

    # def setup_sequence_parallel    
    dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)
    sp_group = dist.new_group(ranks=range(WORLD_SIZE)) ## Set up Ulysses group

    assert WORLD_SIZE % DP == 0, "world size {WORLD_SIZE} not divisible by DP degree {DP}"
    assert WORLD_SIZE % SP == 0, "world size {WORLD_SIZE} not divisible by Ulysses degree {SP}"
    assert hid_dim % hc == 0, "hidden_dim not divisible by hc"
    assert hc % SP == 0, "head count should be divisible by ulysses-degree"

    sub_seq = s // SP
    remainder_sequence = s % SP
    if remainder_sequence != 0: ## if uneven sequence length
        shard_list = [sub_seq + remainder_sequence] + [sub_seq]*(SP-1)
        set_sequence_shard_list(shard_list)

    if RANK == 0: ## first RANK gets extra sequence
        strt_idx = sub_seq * RANK
        end_idx = strt_idx + sub_seq + remainder_sequence 
    else:
        strt_idx = sub_seq * RANK + remainder_sequence
        end_idx = strt_idx + sub_seq
    x_sequence_parallel = x[strt_idx:end_idx, :, :, :].clone() ## prevent memory sharing/aliasing.
    label_sequence_parallel = x[strt_idx:end_idx, :, :, :]

    ## TODO: Add timers (using decorators + labmda?)
    ## Q. Why doesn't time increase as we increase torch.dtype?
    strt = time.time()
    ulysses = DistributedTransformer(embedding_dim, num_layers)
    ulysses_out = ulysses(x_sequence_parallel, sp_group) ## (s/sp, B, hc, hs)
    torch.cuda.synchronize()
    if RANK == 0:
        print(f"total time taken: {time.time() - strt}")

    if unit_test:
        def empty_grad(model):
            for param in model.parameters():
                param.grad = None
        no_ulysses = DistributedTransformer(embedding_dim, num_layers)
        no_ulysses.load_state_dict(ulysses.state_dict()) ## keep the init weights the same

        no_ulysses_out = no_ulysses(x) ## (s, B, hc, hs)

        with torch.no_grad():
            gathered_ulysses_out = torch.empty_like(no_ulysses_out)
            # dist.all_gather_into_tensor(gathered_ulysses_out, ulysses_out, group=sp_group)
            
            out_lst = [torch.empty(local_seq, B, hc, hs) for local_seq in get_sequence_shard_list()]

            # print(f"ulysses_out.shape: {ulysses_out.shape}")
            dist.all_gather(out_lst, ulysses_out, group=sp_group)

            gathered_ulysses_out = torch.cat(out_lst, dim=0) ## TODO: Change dim later? 
            # raise KeyboardInterrupt()

        ## TODO: log max memory.
        # max_mem =
        out_diff = no_ulysses_out - gathered_ulysses_out
        mean_out_diff = torch.norm(out_diff, p=1) / gathered_ulysses_out.numel()
        max_out_diff = out_diff.max()
        # assert mean_out_diff < eps, f"outputs are not close enough. Diff: {mean_out_diff}"
        ## Q. Does the small difference come from ccl or nn layers? 

        no_ulysses_loss = F.mse_loss(no_ulysses_out, label)
        no_ulysses_loss.backward()
        ulysses_loss = F.mse_loss(ulysses_out, label_sequence_parallel)
        ulysses_loss.backward()

        total_grad_diff = 0
        grad_max_diff = 0
        for param1, param2 in zip(no_ulysses.parameters(), ulysses.parameters()):
            diff = param1.grad - param2.grad
            total_grad_diff += torch.norm(diff, p=1)
            grad_max_diff = max(grad_max_diff, diff.max())
        num_parameters = sum([param.numel() for param in ulysses.parameters()])
        mean_grad_diff_per_param = total_grad_diff / num_parameters

        if RANK == 0:
            print(f"Unit Test Results:\n",
                f"out_mean_diff: {mean_out_diff}\n",
                f"out_max_diff: {max_out_diff}", flush=True)

        for i in range(WORLD_SIZE):
            if i == RANK:
                print(f"Rank {RANK}, grad_mean_diff: {mean_grad_diff_per_param}", flush=True)
                print(f"Rank {RANK}, grad_max_diff: {grad_max_diff}", flush=True)
            dist.barrier()

## Learning notes:
## all_to_all: input partitioned list and output gathered list unconcatenated.
## all_to_all_single: scatters the input and concatenates the gathered list for you
## no_ulysses_out.backward(x) ## Interesting use of backward. This is called Jacobian-vector product, 
    # which is mathematically equivalent to summing the output vector into a scalar and computing the gradient 
    # w.r.t. to the input vector (I think).

