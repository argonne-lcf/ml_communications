from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.rank
WORLD_SIZE = comm.size
SP = WORLD_SIZE ## Ulysses_degree
DP = WORLD_SIZE / SP ## TODO: DP TBD

## below needed for CCL? 
# import os
# os.environ['CCL_LOCAL_RANK'] = str(RANK)
# os.environ['CCL_LOCAL_SIZE'] = str(WORLD_SIZE)

assert WORLD_SIZE % DP == 0, "world size {WORLD_SIZE} not divisible by DP degree {DP}"
assert WORLD_SIZE % SP == 0, "world size {WORLD_SIZE} not divisible by Ulysses degree {SP}"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class DistributedMLP(nn.Module):
    def __init__(self, emb_dim, num_layers):
        super().__init__()
        def FFNN(emb_dim=emb_dim):
            ffnn_layer = nn.Sequential(
                nn.Linear(emb_dim, 4*emb_dim),
                nn.GELU(),
                nn.Linear(4*emb_dim, emb_dim),
            )
            return ffnn_layer
        
        self.num_layers = num_layers

        FFNNS = [FFNN() for _ in range(num_layers*2)] ## 2 FFNN per layer
        self.FFNNS = nn.ModuleList(FFNNS)
        
    def forward(self, x, sp_group=None):
        s, B, hc, hs = x.size() ## (s/sp, B, hc, hs) if Ulysses else (s, B, hc, hs)

        for i in range(self.num_layers):
            ## TODO: add QKV projection? 
            ## First all2all: (s/sp, B, hc, hs) -> (s, B, hc/sp, hs)

            if sp_group is not None:
                ## all_to_all: input partitioned list and output gathered list unconcatenated.
                ## all_to_all_single: scatters the input and concatenates the gathered list for you
                full_seq_x = torch.empty(s*SP, B, hc//SP, hs)
                dist.all_to_all_single(full_seq_x, x, group=sp_group)
                x = full_seq_x
                ## Q. how come the DS ulysses did bunch of reshapes and transposes? 

            ## ATTENTION MECHANISM

            ## Second all2all: (s, B, hc/sp, hs) -> (s/sp, B, hc, hs)
            if sp_group is not None:
                partial_seq_x = torch.empty(s, B, hc, hs)
                dist.all_to_all_single(partial_seq_x, x, group=sp_group)
                x = partial_seq_x

            x = x.reshape(-1, B, hc * hs) ## s depends on Ulysses on or off. 
            x = self.FFNNS[2*i](x)
            x = self.FFNNS[2*i+1](x)
            x = x.reshape(-1, B, hc, hs) ## (s/sp, B, hc, hs) if Ulysses else (s, B, hc, hs)

        return x

if __name__ == "__main__":
    ## TODO: add argparse
    B = 1
    s = 144
    hc = 24
    hs = 128
    eps = 1e-6 ## Stingency of gradient and output averaged by num elements. 
    num_layers = 5
    unit_test = True
    ## TODO: add option for different dtypes (half, 8bit)

    ## TODO: select proper backend. 
    dist.init_process_group(backend="", rank=RANK, world_size=WORLD_SIZE)
    # ranks = [for range(RANK * SP + )] ## TODO: make SP-group agnostic to DP
    sp_group = dist.new_group(ranks=range(WORLD_SIZE)) ## Set up Ulysses group
    assert hc % SP == 0, "head count should be divisible by ulysses-degree"

    torch.manual_seed(42) ## sets seed for both CPU and CUDA
    x = torch.randn(s, B, hc, hs)
    label = torch.randn(s, B, hc, hs)
    emb_dim = hc * hs

    ## scatter data across sequence dimension
    assert s % SP == 0, "sequence is not divisible by ulysses-degree"
    sub_seq = s // SP
    strt_idx = sub_seq * RANK
    end_idx = strt_idx + sub_seq
    x_SP = x[strt_idx:end_idx, :, :, :].clone() ## prevent memory sharing/aliasing.

    ## TODO: set devices
    # if torch.cuda.is_available():
    #     dev = 
    # elif torch.xpu.is_available():

    ulysses = DistributedMLP(emb_dim, num_layers)
    ulysses_out = ulysses(x_SP, sp_group) ## (s/sp, B, hc, hs)

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
            dist.all_gather_into_tensor(gathered_ulysses_out, ulysses_out, group=sp_group)

        ## TODO: log max memory. 
        # max_mem = torch.xpu.max_memory_allocated
        mean_out_diff = torch.norm(no_ulysses_out - gathered_ulysses_out, p=1) / gathered_ulysses_out.numel()
        # assert mean_out_diff < eps, f"outputs are not close enough. Diff: {mean_out_diff}"
        ## Q. Does the small difference come from ccl or nn layers? 

        ## TODO: uncomment below after implementing ccl with backward
        # no_ulysses_loss = F.mse_loss(no_ulysses_out, label)
        # no_ulysses_loss.backward()
        # ## TODO: only gather and do compute loss and gradient on rank==0? 
        # ulysses_loss = F.mse_loss(ulysses_out, label)
        # ulysses_loss.backward()
        # # no_ulysses_out.backward(x) ## Interesting use of backward. This is called Jacobian-vector product, 
        # # which is mathematically equivalent to summing the output vector into a scalar and computing the gradient 
        # # w.r.t. to the input vector (I think).

        # out_diff = torch.norm(no_ulysses_out - ulysses_out, p=1) ## l1 norm
        # total_grad_diff = 0
        # for grad1, grad2 in zip(no_ulysses.parameters(), ulysses.parameters()):
        #     total_grad_diff += torch.norm(grad1 - grad2, p=1)
        # num_parameters = sum([param.numel() for param in ulysses.parameters()])
        # mean_grad_diff_per_param = total_grad_diff / num_parameters

        if RANK == 0:
            print(f"out_diff: {mean_out_diff}")
            # print(f"total_grad_diff: {total_grad_diff}")
            # print(f"mean_grad_diff_per_param: {mean_grad_diff_per_param}")

        # assert out_diff < eps
        # assert mean_grad_diff_per_param < eps