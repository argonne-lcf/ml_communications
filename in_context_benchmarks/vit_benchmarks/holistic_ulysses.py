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

## TBD
# class All2All(torch.autograd.Function):
#     def forward(ctx, gather, scatter):


#     def backward():

# def all2all_single():

class DistributedMLP(nn.Module):
    def __init__(self, emb_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.get_qkvs = nn.ModuleList(nn.Linear(emb_dim, 3*emb_dim) for _ in range(num_layers)) ## Fused qkv layer w/ bias term
        self.dense_outs = nn.ModuleList(nn.Linear(emb_dim, emb_dim) for _ in range(num_layers)) ## dense layer on att_out
        
    def forward(self, x:torch.tensor, sp_group=None, comm_only:bool=False) -> torch.tensor:
        loc_s, B, hc, hs = x.size()
        for get_qkv, dense_out in zip(self.get_qkvs, self.dense_outs):
            x = x.view(loc_s, B, hc*hs) ## [loc_s, B, hc*hs]: flatten to perform full GEMM
            qkv = get_qkv(x) ## [loc_s, B, 3*hc*hs] - project to 3xdim for qkv
            
            ## First all2all: 
            if sp_group is not None:
                qkv = (
                    qkv
                    ## Q. How .view works: would it matter if SP was on the right of hc//sp?
                    .view(loc_s, B, SP, hc//SP, 3*hs) ## split head by SP degree
                    .permute(2, 0, 1, 3, 4) ## [sp, loc_s, B, hc/sp, 3*hs]
                    .contiguous()
                )
                ## Q. First dim have to be WORLD_SIZE? Why? dist all2all is VERY mysterious.
                first_all2lall = torch.empty_like(qkv)
                dist.all_to_all_single(first_all2lall, qkv, group=sp_group)
                qkv = first_all2lall.reshape(loc_s*SP, B, hc//SP, 3*hs) ## [s, B, hc/sp, 3*hs]
            else:
                qkv = qkv.view(loc_s, B, hc, 3*hs)

            ## ATTENTION MECHANISM
            qkv = qkv.permute(1, 2, 0, 3) ## [B, hc/sp, loc_s, 3*hs] Follow torch attention's syntax
            q, k, v = qkv.tensor_split(3, dim=-1) ## [B, hc/sp, loc_s, hs] x3
            att_out = F.scaled_dot_product_attention(q, k, v) ## [B, hc/sp, loc_s, hs] x3 -> [B, hc/sp, s, hs]
            x = att_out.permute(2, 0, 1, 3) ## [s, B, hc/sp, hs] - Follow all2all syntax

            ## Second all2all: 
            if sp_group is not None:
                x = x.reshape(SP, loc_s, B, hc//SP, hs) ## extract SP
                second_all2all = torch.empty_like(x)
                dist.all_to_all_single(second_all2all, x, group=sp_group)
                x = (
                    second_all2all
                    .permute(1, 2, 0, 3, 4) ## [loc_s, B, sp, hc/sp, hs]
                    .reshape(loc_s, B, hc, hs) ## [loc_s, B, hc, hs]
                )
            x = x.view(loc_s, B, hc*hs)
            x = dense_out(x)
            x = x.view(loc_s, B, hc, hs)
        return x

if __name__ == "__main__":
    ## TODO: add argparse
    # import argparse
    B = 1
    s = 4096
    hid_dim = 4096
    hc = 16
    hs = hid_dim // hc
    num_layers = 24
    unit_test = True
    dtype = torch.float32 ## {torch.bfloat16, torch.float16, etc}

    ## Global setting:
    # torch.backends.cuda.enable_flash_sdp(enabled) ## globally enable FA2. By default, let pytorch choose the most optimal one
    torch.set_default_device(RANK)
    torch.set_default_dtype(dtype)

    dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)
    ## TODO: make SP-group agnostic to DP
    sp_group = dist.new_group(ranks=range(WORLD_SIZE)) ## Set up Ulysses group
    
    assert WORLD_SIZE % DP == 0, "world size {WORLD_SIZE} not divisible by DP degree {DP}"
    assert WORLD_SIZE % SP == 0, "world size {WORLD_SIZE} not divisible by Ulysses degree {SP}"
    assert hid_dim % hc == 0, "hidden_dim not divisible by hc"
    assert hc % SP == 0, "head count should be divisible by ulysses-degree"
    assert s % SP == 0, "sequence is not divisible by ulysses-degree"

    torch.manual_seed(42) ## seed for both CPU and CUDA
    x = torch.randn(s, B, hc, hs)
    label = torch.randn(s, B, hc, hs)
    emb_dim = hc * hs

    ## scatter data across sequence dimension
    sub_seq = s // SP
    strt_idx = sub_seq * RANK
    end_idx = strt_idx + sub_seq
    x_SP = x[strt_idx:end_idx, :, :, :].clone() ## prevent memory sharing/aliasing.

    ## TODO: Add thorough timers
    ## Q. why doesn't this increase as we increase torch.dtype?
    strt = time.time()
    ulysses = DistributedMLP(emb_dim, num_layers)
    ulysses_out = ulysses(x_SP, sp_group) ## (s/sp, B, hc, hs)
    torch.cuda.synchronize() ## Q. can't we just use barrier?
    print(time.time() - strt)

    ## TODO: Make the unit_test separate?
    if unit_test:
        def empty_grad(model):
            for param in model.parameters():
                param.grad = None
        no_ulysses = DistributedMLP(emb_dim, num_layers)
        no_ulysses.load_state_dict(ulysses.state_dict()) ## keep the init weights the same

        no_ulysses_out = no_ulysses(x) ## (s, B, hc, hs)

        ## TODO: Implement custom ccl with backpropagation
        with torch.no_grad():
            gathered_ulysses_out = torch.empty_like(no_ulysses_out)
            ## Q. Does regular all-gather also have gather along only 1st dimension constarint?
            dist.all_gather_into_tensor(gathered_ulysses_out, ulysses_out, group=sp_group)

        ## TODO: log max memory.
        # max_mem =
        mean_out_diff = torch.norm(no_ulysses_out - gathered_ulysses_out, p=1) / gathered_ulysses_out.numel()
        # assert mean_out_diff < eps, f"outputs are not close enough. Diff: {mean_out_diff}"
        ## Q. Does the small difference come from ccl or nn layers? 

        ## TODO: uncomment below after implementing backward
        # no_ulysses_loss = F.mse_loss(no_ulysses_out, label)
        # no_ulysses_loss.backward()
        # ## TODO: only gather and do compute loss and gradient on rank==0? 
        # ulysses_loss = F.mse_loss(ulysses_out, label)
        # ulysses_loss.backward()

        # out_diff = torch.norm(no_ulysses_out - ulysses_out, p=1) ## l1 norm
        # total_grad_diff = 0
        # for grad1, grad2 in zip(no_ulysses.parameters(), ulysses.parameters()):
        #     total_grad_diff += torch.norm(grad1 - grad2, p=1)
        # num_parameters = sum([param.numel() for param in ulysses.parameters()])
        # mean_grad_diff_per_param = total_grad_diff / num_parameters

        if RANK == 0:
            print(f"mean_out_diff: {mean_out_diff}")
            # print(f"total_grad_diff: {total_grad_diff}")
            # print(f"mean_grad_diff_per_param: {mean_grad_diff_per_param}")

        # assert out_diff < eps
        # assert mean_grad_diff_per_param < eps

## Learning notes:
## all_to_all: input partitioned list and output gathered list unconcatenated.
## all_to_all_single: scatters the input and concatenates the gathered list for you
## no_ulysses_out.backward(x) ## Interesting use of backward. This is called Jacobian-vector product, 
    # which is mathematically equivalent to summing the output vector into a scalar and computing the gradient 
    # w.r.t. to the input vector (I think).

