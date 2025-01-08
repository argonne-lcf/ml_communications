import os
import sys
import socket
import numpy as np
import time
from typing import Union, Optional
from argparse import ArgumentParser
from time import perf_counter
import random

try:
    #import mpi4py
    #mpi4py.rc.initialize = False
    from mpi4py import MPI
    WITH_DDP = True
except ModuleNotFoundError as e:
    WITH_DDP = False
    pass

import torch
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError as e:
    pass
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn as distnn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import oneccl_bindings_for_pytorch as ccl
except ModuleNotFoundError as e:
    pass

TORCH_FLOAT_DTYPE = torch.float32

# Get MPI:
if WITH_DDP:
    LOCAL_RANK = int(os.getenv("PALS_LOCAL_RANKID"))
    #LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    COMM = MPI.COMM_WORLD

    try:
        WITH_CUDA = torch.cuda.is_available()
        if RANK == 0: print('Running on CUDA devices',flush=True)
    except:
        WITH_CUDA = False
        pass

    try:
        WITH_XPU = torch.xpu.is_available()
        if RANK == 0: print('Running on XPU devices',flush=True)
    except:
        WITH_XPU = False
        pass

    if WITH_CUDA:
        DEVICE = torch.device('cuda')
        N_DEVICES = torch.cuda.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES>1 else 0
        torch.cuda.set_device(DEVICE_ID)
    elif WITH_XPU:
        DEVICE = torch.device('xpu')
        N_DEVICES = torch.xpu.device_count()
        DEVICE_ID = LOCAL_RANK if N_DEVICES>1 else 0
        torch.xpu.set_device(DEVICE_ID)
    else:
        DEVICE = torch.device('cpu')
        DEVICE_ID = 'cpu'
        if RANK == 0: print('Running on CPU devices',flush=True)
else:
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    print('MPI Initialization failed!',flush=True)


def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)
    elif WITH_XPU:
        backend = 'ccl' if backend is None else str(backend)
    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )


def cleanup() -> None:
    dist.destroy_process_group()


def get_neighbors(args):
    neighbors = []
    if 'optimized' in args.all_to_all_buff:
        if SIZE==1:
            neighbors = [0]
        else:
            rank_list = [i for i in range(SIZE)]
            if args.neighbors == 'random':
                while len(neighbors)<args.num_neighbors:
                    rank = random.choice(rank_list)
                    if rank not in neighbors and rank!=RANK:
                        neighbors.append(rank)
            elif args.neighbors == 'nearest':
                for i in range(args.num_neighbors):
                    left_rank = (RANK-(1+i))%SIZE
                    right_rank = (RANK+(1+i))%SIZE
                    neighbors.extend([left_rank, right_rank])
        print(f'[{RANK}] neighbor list: {neighbors}',flush=True)
    return neighbors


def build_buffers(args, neighbors):
    buff_send_sz = [0] * SIZE
    buff_recv_sz = [0] * SIZE

    if args.all_to_all_buff == 'naive':
        buff_send = [torch.empty(0, device=DEVICE)] * SIZE
        buff_recv = [torch.empty(0, device=DEVICE)] * SIZE
        for i in range(SIZE):
            buff_send[i] = torch.empty([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_send_sz[i] = torch.numel(buff_send[i])*buff_send[i].element_size()/1024/1024
            buff_recv[i] = torch.empty([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_recv_sz[i] = torch.numel(buff_recv[i])*buff_recv[i].element_size()/1024/1024
    elif args.all_to_all_buff == 'optimized':
        buff_send = [torch.empty(0, device=DEVICE)] * SIZE
        buff_recv = [torch.empty(0, device=DEVICE)] * SIZE
        for i in neighbors:
            buff_send[i] = torch.empty([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_send_sz[i] = torch.numel(buff_send[i])*buff_send[i].element_size()/1024/1024
            buff_recv[i] = torch.empty([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_recv_sz[i] = torch.numel(buff_recv[i])*buff_recv[i].element_size()/1024/1024
    elif args.all_to_all_buff == 'semi-optimized':
        #buff_send = [torch.empty([], device=DEVICE)] * SIZE
        #buff_recv = [torch.empty([], device=DEVICE)] * SIZE
        buff_send = [torch.zeros(1, device=DEVICE)] * SIZE
        buff_recv = [torch.zeros(1, device=DEVICE)] * SIZE
        for i in neighbors:
            #buff_send[i] = torch.empty([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_send[i] = torch.zeros([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_send_sz[i] = torch.numel(buff_send[i])*buff_send[i].element_size()/1024/1024
            #buff_recv[i] = torch.empty([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_recv[i] = torch.zeros([args.num_nodes, args.num_features], dtype=TORCH_FLOAT_DTYPE, device=DEVICE)
            buff_recv_sz[i] = torch.numel(buff_recv[i])*buff_recv[i].element_size()/1024/1024

    # Print information about the buffers
    print('[RANK %d]: Created send and receive buffers for %s halo exchange:' %(RANK,args.all_to_all_buff), flush=True)
    print(f'[RANK {RANK}]: Send buffers of size [MB]: {buff_send_sz}',flush=True)
    print(f'[RANK {RANK}]: Receive buffers of size [MB]: {buff_recv_sz}',flush=True)

    return [buff_send, buff_recv]


def halo_test(args, neighbors, buffers) -> None:
    buff_send_safe = buffers[0]
    buff_recv_safe = buffers[1]
    buff_send = buff_send_safe
    buff_recv = buff_recv_safe

    times = []
    for itr in range(args.iterations):
        # initialize the buffers 
        for i in range(SIZE):
            buff_send[i] = torch.empty_like(buff_send_safe[i])
            buff_recv[i] = torch.empty_like(buff_recv_safe[i])
        
        # fill in the non-empty buffers
        if args.all_to_all_buff == 'naive':
            for i in range(SIZE):
                buff_send[i] = torch.randn_like(buff_send[i])
        elif 'optimized' in args.all_to_all_buff:
            for i in neighbors:
                buff_send[i] = torch.randn_like(buff_send[i])

        # Perform the all_to_all
        tic = perf_counter()
        distnn.all_to_all(buff_recv, buff_send)
        if WITH_CUDA:
            torch.cuda.synchronize()
        elif WITH_XPU:
            torch.xpu.synchronize()
        toc = perf_counter()
        times.append(toc-tic)

    # Get stats, remove first two iterations
    #if RANK==0: print(times)
    #times.pop(0)
    #times.pop(0)
    # For Aurora, need to throw away first 10 iterations...
    times = times[10:]
    avg_time = sum(times)/len(times)
    return avg_time
    

def main() -> None:
    # Say hi
    print(f'Hello from rank {RANK}/{SIZE}, local rank {LOCAL_RANK}, on device {DEVICE}:{DEVICE_ID} out of {N_DEVICES}', flush=True)

    # Parse arguments
    parser = ArgumentParser(description='GNN for ML Surrogate Modeling for CFD')
    parser.add_argument('--all_to_all_buff', default='naive', type=str, choices=['naive','optimized','semi-optimized'], help='Type of all_to_all buffers')
    parser.add_argument('--num_nodes', default=32768, type=int, help='Number of input nodes (rows) to the all_to_all')
    parser.add_argument('--num_features', default=8, type=int, help='Number of input features (columns) to the all_to_all')
    parser.add_argument('--num_neighbors', default=1, type=int, help='Number of neighbors involved in the all_to_all')
    parser.add_argument('--neighbors', default='nearest', type=str, choices=['nearest','random'], help='Strategy for gathering neighbors')
    parser.add_argument('--iterations', default=20, type=int, help='Number of iterations to run')
    parser.add_argument('--master_addr', default=None, type=str, help='Master address for torch.distributed')
    parser.add_argument('--master_port', default=None, type=int, help='Master port for torch.distributed')
    args = parser.parse_args()
    if args.neighbors=='random':
        assert args.num_neighbors<=SIZE, 'Number of neighbors must be less than or equal to the number of ranks'
    elif args.neighbors=='nearest':
        if SIZE>1:
            assert args.num_neighbors*2<=SIZE, 'Number of neighbors x 2 must be less than or equal to the number of ranks'
    if RANK == 0:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('RUNNING WITH INPUTS:')
        print(args)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n',flush=True)

    if WITH_DDP:
        os.environ['RANK'] = str(RANK)
        os.environ['WORLD_SIZE'] = str(SIZE)
        if args.master_addr is not None:
            MASTER_ADDR = str(args.master_addr) if RANK == 0 else None
        else:
            MASTER_ADDR = socket.gethostname() if RANK == 0 else None
        MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        if args.master_port is not None:
            os.environ['MASTER_PORT'] = str(args.master_port)
        else:
            os.environ['MASTER_PORT'] = str(2345)
        init_process_group(RANK, SIZE)
        
        random.seed(42+int(RANK))
        neighbors = get_neighbors(args)
        buffers = build_buffers(args, neighbors)
        COMM.Barrier()
        avg_time = halo_test(args, neighbors, buffers)
        if RANK == 0:
            print(f'\n\nAverage all2all time: {avg_time:>4e} sec',flush=True)        

        cleanup()


if __name__ == '__main__':
    main()
