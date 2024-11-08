from mpi4py import MPI
comm = MPI.COMM_WORLD
RANK = comm.rank
WORLD_SIZE = comm.size
SP = WORLD_SIZE ## Ulysses_degree
DP = WORLD_SIZE // SP ## TODO: DP TBD

import torch
import torch.distributed as dist
import time

if __name__ == "__main__":
    ## TODO: add argparse
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
    sp_group = dist.new_group(ranks=range(WORLD_SIZE)) ## Ulysses group
    
    assert WORLD_SIZE % DP == 0, "world size {WORLD_SIZE} not divisible by DP degree {DP}"
    assert WORLD_SIZE % SP == 0, "world size {WORLD_SIZE} not divisible by Ulysses degree {SP}"
    assert hid_dim % hc == 0, "hidden_dim not divisible by hc"
    assert hc % SP == 0, "head count should be divisible by ulysses-degree"
    assert s % SP == 0, "sequence is not divisible by ulysses-degree"

    torch.manual_seed(42) ## seed for both CPU and CUDA
    qkv = torch.randn(s//SP, B, hc, 3*hs)
    att_out = torch.randn(s//SP, B, hc, hs)

    ## pure comm (forward)
    for l in range(num_layers):
        strt = time.time()
        first_all2all = torch.empty_like(qkv)
        dist.all_to_all_single(first_all2all, qkv, group=sp_group)
        first_all2all_time = time.time()

        second_all2all = torch.empty_like(att_out)
        dist.all_to_all_single(second_all2all, att_out, group=sp_group)
        second_all2all_time = time.time()

        if RANK == 0: ## TODO: get the max of comm time?
            print(f"layer{l}_fwd first all2all time: {first_all2all_time - strt:3f}", flush=True)
            print(f"layer{l}_fwd second all2all time: {second_all2all_time - strt:3f}", flush=True)

    ## pure comm (bwd)
    for l in range(num_layers):
        strt = time.time()
        second_all2all = torch.empty_like(att_out)
        dist.all_to_all_single(second_all2all, att_out, group=sp_group)
        second_all2all_time = time.time()

        first_all2all = torch.empty_like(qkv)
        dist.all_to_all_single(first_all2all, qkv, group=sp_group)
        first_all2all_time = time.time()

        if RANK == 0: ## TODO: get the max of comm time?
            print(f"layer{l}_bwd second all2all time: {second_all2all_time - strt:3f}", flush=True)
            print(f"layer{l}_bwd first all2all time: {first_all2all_time - strt:3f}", flush=True)




