import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser(description="parse input arguments for the gpu allreduce benchmark")

parser.add_argument("-dim", "--tensor_dimension_1d",
                        help="The size of the 1d tensor that is distributed accross the ranks per node.",
                        type=int, default=4096)
args = parser.parse_args()

def main(tensor_dimension_1d):
    t1 = perf_counter_ns() 
    #import intel_extension_for_pytorch  # Added Extra
    import torch
    import torch.nn.parallel
    import torch.distributed as dist
    #import oneccl_bindings_for_pytorch
    t2 = perf_counter_ns() 
    import_timer = t2 - t1

    MPI.COMM_WORLD.Barrier()

    os.environ['RANK']          = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE']    = str(os.environ.get('PMI_SIZE', 1))
    mpi_world_size              = MPI.COMM_WORLD.Get_size()
    mpi_my_rank                 = MPI.COMM_WORLD.Get_rank()

    if mpi_my_rank == 0:
        master_addr              = socket.gethostname()
        sock                     = socket.socket()
        sock.bind(('',0))
        # master_port  = sock.getsockname()[1] 
        master_port              = 2345
    else:
        master_addr              = None
        master_port              = None

    master_addr                 = MPI.COMM_WORLD.bcast(master_addr, root=0)
    master_port                 = MPI.COMM_WORLD.bcast(master_port, root=0)
    os.environ["MASTER_ADDR"]   = master_addr
    os.environ["MASTER_PORT"]   = str(master_port)

    MPI.COMM_WORLD.Barrier()
    t3 = perf_counter_ns() 
    dist.init_process_group(backend = "nccl", init_method = 'env://', world_size = mpi_world_size, rank = mpi_my_rank, timeout = datetime.timedelta(seconds=3600))
    t4 = perf_counter_ns() 
    init_timer = t4 - t3
    MPI.COMM_WORLD.Barrier()


    dist_my_rank        = dist.get_rank()
    dist_world_size     = dist.get_world_size()
    device_count = torch.cuda.device_count()

    def get_default_device():
        if torch.cuda.is_available():
            #return torch.device(f"cuda:{dist_my_rank%4}")
            return torch.device(f"cuda:{dist_my_rank%int(device_count)}")
        else:
            return torch.device('cpu')

    device  = get_default_device()

    #dim_size=int(int(sys.argv[1])/4)
    dim_size=int(int(tensor_dimension_1d)/4)
    MPI.COMM_WORLD.Barrier()

    elapsed1=[]
    total_elapsed=0.0

    for _ in range(10):
        x = torch.ones([1, dim_size],dtype=torch.float32).to(device, non_blocking=True)
        # print(x)
        t5 = perf_counter_ns() 
        dist.all_reduce(x, op=dist.ReduceOp.SUM)  # Added Extra op
        #MPI.COMM_WORLD.Barrier()
        torch.cuda.synchronize()
        t6 = perf_counter_ns()
        elapsed1.append(t6 - t5)
        total_elapsed += (t6-t5)

    if mpi_my_rank == 0:
        print(f"Python Import time = {import_timer / 1000 / 1000 / 1000} s")
        print(f"DDP initialization time = {init_timer / 1000 / 1000 / 1000} s")
        print(f"Message size = {(dim_size * 4) / 1024 / 1024} MB")
        print(f"Total time = {total_elapsed / 1000 / 1000 / 1000} s")
        for idx, e in enumerate(elapsed1):
            if idx==0:
                print(f"ALLREDUCE {idx} took {e / 1000 / 1000 / 1000} s")
            else:
                print(f"ALLREDUCE {idx} took {e / 1000 / 1000} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse input arguments for the gpu allreduce benchmark")

    parser.add_argument("-dim", "--tensor_dimension_1d",
                        help="The size of the 1d tensor that is distributed accross the ranks per node.",
                        type=int, default=4096)
    args = parser.parse_args()
    
    main(args.tensor_dimension_1d)



