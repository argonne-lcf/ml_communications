"""
Partial benchmark focusing on the compute and communication pattern of
sequence parallelism.

Note: Here all the operations overwrite the input tensor. The input is, defined
as the input per rank. The matrix shapes are chosen to mimic a particular 
implementation of sequence parallelism in Megatron-DeepSpeed. 

Important Variables:
    - d1, S: sequence length
    - d2, H: number of hidden dimension
    - precision type: float32
    - mm1: weight matrix to be applied before the attention layer
    - mm2: weight matrix to be applied after the attention layer
    - time0: total time taken for the first all_gather in the timing loop
    - time1: total time taken for the application of the first weight matrix
    - time2: total time taken for the application of the second wight matrix
    - time3: total time taken for the reduce scatter operation
    - inmean0: mean value of the inputs
    - inmean2: mean value of the results

Pattern of compute operations:
    - Intialize data on available ranks on a node
    - Perform an all_gather
    - Apply weight matrix
    - Apply attention matrix (to be implemented or come from DeepSpeed)
    - Apply weight matrix after the attention
    - Perform a reduce scatter of the result

Data and Input shapes:
    - All Gather Buffer (A) shape: (S, 1, H) == (d1, 1, d2)
    - Input per rank: (S/NGPUS, 1, H) == (d1//12, 1, d2)
    - Weight matrix 1, mm1: (H//12, H) == (d2//12, d1)
    - Weight matrix 2, mm2: (H, H//12) == (d1, d2//12)
    - Result after application of mm1, A*mm1^T == (S, 1, H//12) == (d1, 1, d2//12)
    - Result after application of mm2, A*mm1^T*mm2^T == (S, 1, H) == (d1, 1, d2)
"""
#print("being import", flush=True)
from mpi4py import MPI
import os
import time
import socket
import argparse
from collections import namedtuple
import logging

import torch
import numpy as np

#from timer_scripts import timer, timer_gpu 

#import intel_extension_for_pytorch as ipex  # noqa: F401 # type: ignore
#import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
#print("being code", flush=True)

parser = argparse.ArgumentParser(description="parse input arguments for sequence parallel partial benchmark")

parser.add_argument("-s", "--sequence_length", 
                    help="Maximum sequence length. The size of the ALLGATHER buffer", 
                    type=int, default=4608)
parser.add_argument("-d", "--hidden_dimension", 
                    help="Hidden dimension for the matrix multiplication. Proxy for the model size.", 
                    type=int, default=9216)
parser.add_argument("-it", "--iterations", 
                    help="number of iterations for the timing loop", 
                    type=int, default=18)
parser.add_argument("-wit", "--warmup_iterations", help="number of warmup iterations", 
                    type=int, default=10)
parser.add_argument("-p", "--precision", help="Data type for the elements of a tensor. float32 and bfloat16 supported.", 
                    type=str, default="float32")

parser.add_argument("-dvc", "--device", help="Device type. cuda and xpu supported.", 
                    type=str, default="cuda")

parser.add_argument("-f", "--log_file", help="Output file name", 
                    type=str, default="sequence_parallel.log")

parser.add_argument("-dir", "--log_directory", help="Output file path", 
                    type=str, default="logs/")

parser.add_argument("--logging", help="Switch logging on", action='store_true')

args = parser.parse_args()

log_directory = args.log_directory  # Directory for logs
log_filename = args.log_file  # Log file name
log_filepath = os.path.join(log_directory, log_filename)

if args.device == "xpu":
    import intel_extension_for_pytorch as ipex 
    import oneccl_bindings_for_pytorch 

def synchronize(device):
    if device == "cuda":
        return torch.cuda.synchronize()
    elif device == "xpu":
        return torch.xpu.synchronize()
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

def get_device_string(device):
    if device == "cuda": 
        return f"cuda:{torch.cuda.current_device()}" 
    elif device == "xpu":
        
        return f"xpu:{torch.xpu.current_device()}" 
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

def get_device_count(device):
    if device == "cuda":
        return torch.cuda.device_count() 
    elif device == "xpu":
        return torch.xpu.device_count() 
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

def set_device(visible_device):
    if args.device == "cuda":
        return torch.cuda.set_device(visible_device)
    elif args.device == "xpu":
        return torch.xpu.set_device(visible_device)
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

def get_backend(device):
    if device =="cuda":
        return "nccl"
    elif device == "xpu":
        return "ccl"
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

rank = int(MPI.COMM_WORLD.Get_rank())
world_size = int(MPI.COMM_WORLD.Get_size())

if args.logging:
    if rank == 0 and not os.path.exists(log_directory):
        # Create log directory if it doesn't exist
        os.makedirs(log_directory)

    MPI.COMM_WORLD.Barrier()
    logging.basicConfig(filename=log_filepath, filemode="a", level="INFO")
else:
    logging.basicConfig(level="INFO")

#print(f"rank {rank}/{world_size}")
logging.info(f"rank {rank}/{world_size}")

visible_device = rank % get_device_count(args.device)
local_rank = visible_device
set_device(visible_device)

if rank == 0:
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

torch.distributed.init_process_group(
    backend=get_backend(args.device),
    init_method="env://",
    world_size=world_size,
    rank=rank,
)

if args.precision == "float32":
    data_type = torch.float32
    data_type_multiplier = 32 ## 32 Bits = 4 Bytes
elif args.precision == "bfloat16":
    data_type = torch.bfloat16
    data_type_multiplier = 16 ## 16 Bits

S = args.sequence_length #4608 #4608 sequence length
H = args.hidden_dimension #9216 #9216 hidden dimension

all_gather_buffer = torch.zeros([S, 1, H], dtype=data_type, device=get_device_string(args.device))
input = torch.rand([S//world_size, 1, H], dtype=data_type, device=get_device_string(args.device))

attn_W_QKV = torch.rand(
        H//world_size,
        H,
        device=get_device_string(args.device),
        dtype=data_type,
    )*1e-10
attn_W_0 = torch.rand(
        H,
        H//world_size,
        device=get_device_string(args.device),
        dtype=data_type,
    )*1e-8

input_mean_before_operations=input.mean()

def sequence_parallel(iterations, warmup=False):
    t_allgather = 0.0
    t_W_QKV = 0.0
    t_W_0 = 0.0
    t_reduce_scatter = 0.0

    list_all_gather_times = np.zeros(args.iterations)
    list_reduce_scatter_times = np.zeros(args.iterations)
    SequenceParallelResults = namedtuple("SequenceParallelResults", 
                      ["t_allgather", "t_W_QKV", "t_W_0", "t_reduce_scatter", 
                       "list_all_gather_times", "list_reduce_scatter_times"])

    for i in range(iterations):
        start = time.perf_counter_ns()
        torch.distributed.all_gather_into_tensor(
            all_gather_buffer, input
        )
        synchronize(args.device)
        end = time.perf_counter_ns()
        if warmup:
            if rank == 0:
                logging.info("Doing Warmups")
            t_allgather =  0.0
        else:
            t_allgather += end-start
            list_all_gather_times[i] = (end-start)
            start = end

        intermediate = torch.matmul(all_gather_buffer, attn_W_QKV.t())
        synchronize(args.device)
        end = time.perf_counter_ns()
        if warmup:
            t_W_QKV = 0.0
        else:
            t_W_QKV += end-start
            #FA would be here
            start = end
        intermediate = torch.matmul(intermediate, attn_W_0.t())
        synchronize(args.device)
        end = time.perf_counter_ns()
        if warmup:
            t_W_0 = 0.0
        else:
            t_W_0 += end-start
            start = end
        torch.distributed.reduce_scatter_tensor(
            input, intermediate
        )
        synchronize(args.device)
        end = time.perf_counter_ns()
        if warmup:
            t_reduce_scatter = 0.0
        else:
            t_reduce_scatter += end-start
            list_reduce_scatter_times[i] = (end-start)
        #gather optimizer states
        #allreduce model updates
    return SequenceParallelResults(t_allgather, t_W_QKV, t_W_0, t_reduce_scatter, 
                                   list_all_gather_times, list_reduce_scatter_times)


# Doing Warmups
sequence_parallel(iterations=args.warmup_iterations, warmup=True)
# Collecting results
start_time=time.perf_counter_ns()
Results = sequence_parallel(iterations=args.iterations, warmup=False)
end_time=time.perf_counter_ns()
# Extracting results
total_time_allgather = Results.t_allgather
total_time_W_QKV = Results.t_W_QKV
total_time_W_0 = Results.t_W_0
total_time_reduce_scatter = Results.t_reduce_scatter
all_gather_times = Results.list_all_gather_times 
reduce_scatter_times = Results.list_reduce_scatter_times

if rank == 0:
    logging.info(f"\n ==== Main Results ==== \n")
    logging.info(f"Running with {args.precision} data type")
    logging.info(f"\n ==== List of Arguments ==== \n")
    logging.info(f"Sequence Length = {args.sequence_length}")
    logging.info(f"Hidden Dimension = {args.hidden_dimension}")
    logging.info(f"Precision Type = {args.precision}")
    logging.info("\n ==== List of Arguments ==== \n")
    logging.info(f"Input shape = {input.shape}")
    logging.info(f"Matrix W_QKV  shape = {attn_W_QKV.shape}")
    logging.info(f"Matrix W_0 shape = {attn_W_0.shape}")
    logging.info(f"ALLGATHER Buffer size = {(args.sequence_length * data_type_multiplier) / 8 / 1024 / 1024} MB")
    #print("times", time0*1000, time1*1000, time2*1000, time3*1000)
    logging.info(f"Mean before all operations = {input_mean_before_operations}")
    logging.info(f"Total time taken for ALLGATHER = {total_time_allgather / 1e6} ms" )
    logging.info(f"Total time taken for matrix multiplication 1 = {total_time_W_QKV / 1e6} ms")
    logging.info(f"Total time taken for matrix multiplication 2 = {total_time_W_0 / 1e6} ms")
    logging.info(f"Total time taken for REDUCE_SCATTER = {total_time_reduce_scatter / 1e6} ms")
    logging.info(f"total time for the loop = {(end_time-start_time) / 1e6} ms")
    logging.info(f"Mean after all operations = {input.mean()}")
    logging.info(f"ALLGATHER Throughput = {(args.sequence_length * data_type_multiplier * args.iterations) / (total_time_allgather / 1e9) / 8 / 1024 / 1024 } MB/s")
    logging.info(f"REDUCE_SCATTER Throughput = {(args.sequence_length * data_type_multiplier * args.iterations) / (total_time_reduce_scatter / 1e9) / 8 / 1024 / 1024 } MB/s")

    for idx, (t_ag, t_rs) in enumerate(zip(all_gather_times, reduce_scatter_times)):
        logging.info(f"ALLGATHER {idx} takes {t_ag / 1e6} ms, REDUCE_SCATTER {idx} takes {t_rs / 1e6} ms")
    logging.info(f"\n ==== Finished Running ==== \n")
