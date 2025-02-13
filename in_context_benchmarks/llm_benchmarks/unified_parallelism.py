from mpi4py import MPI
import os
import time
import math
import socket
import argparse
import logging
from collections import namedtuple

import torch
import numpy as np
import torch.distributed
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist

#import intel_extension_for_pytorch as ipex  # noqa: F401 # type: ignore
#import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore

parser = argparse.ArgumentParser(description="parse input arguments for tensor and data  parallel partial benchmark")
parser.add_argument("-s", "--sequence_length",
                    help="Maximum sequence length. The size of the ALLGATHER buffer",
                    type=int, default=4608)
parser.add_argument("-d", "--hidden_dimension",
                    help="Hidden dimension for the matrix multiplication. Proxy for the model size.",
                    type=int, default=9216)
parser.add_argument("-it", "--iterations",
                    help="number of iterations for the timing loop",
                    type=int, default=3)
parser.add_argument("-wit", "--warmup_iterations", 
                    help="number of warmup iterations",
                    type=int, default=2)
parser.add_argument("-bucket", "--grad_allreduce_bucket", 
                    help="The largest bucket of message passed to do the allreduce over the data parallel groups (in number of elements in a message).",
                    type=int, default=5e8)
parser.add_argument("-dp_switch", "--data_parallel_switch", 
                    help="If TRUE, calculates data parallelism degree based of tp_degree",
                    type=bool, default=True)
parser.add_argument("-tp_degree", "--tensor_parallel_degree", 
                    help="Tensor Parallel degree. In this context, the model is distributed across the number of (tensor parallel degree)) ranks",
                    type=int, default=None)
parser.add_argument("-sp_switch", "--sequence_parallel_switch", 
                    help="Switch sequence parallelism on or off", action='store_true')
parser.add_argument("-n_layers", "--number_of_transformer_layers", 
                    help="Number of transformer layers", type=int, default=80)
parser.add_argument("--init_std", 
                    help="Standard deviation for initializing weights and inputs with normal distribution",
                    type=float, default=0.01)
parser.add_argument("--init_mean", 
                    help="Mean for initializing weights and inputs with normal distribution",
                    type=float, default=0.0)
parser.add_argument("-p", "--precision", 
                    help="Data type for the elements of a tensor. float32 and bfloat16 supported.",
                    type=str, default="float32")
parser.add_argument("-dvc", "--device", help="Device type. cuda and xpu supported.",
                    type=str, default="cuda")
parser.add_argument("-f", "--log_file", help="Output file name",
                    type=str, default="tensor_parallel.log")
parser.add_argument("-dir", "--log_directory", help="Output file path",
                    type=str, default="logs/")
parser.add_argument("--logging", help="Switch logging on", action='store_true')
parser.add_argument("--save", help="Save detail results in npy format", action='store_true') ## Generates huge files, use with caution
parser.add_argument("--trace", default=None, type=str)

args = parser.parse_args()

def trace_func(func):
   def wrapper(*args, **kwargs):
      try:
         function_name = func.__func__.__qualname__
      except:
         function_name = func.__qualname__
      with record_function(function_name):
         return func(*args, **kwargs)
   return wrapper

@trace_func
def tensor_parallel(N_timing_loop, n_layers, n_iter_grad_sync, 
                    input, partial_input, partial_interim2, 
                    partial_interim4, attn_W_QKV, attn_WO, 
                    mat_h_4h, mat_4h_h, allreduce_grad, warmup=False):
    # Define operations based on parallelisms
    operations = ["QKV", "WO", "H_4H", "4H_H", "grad_sync", "timing_loop"]
    if SP:
        operations += ["allgather_1", "allgather_2", "reduce_scatter_1", "reduce_scatter_2"]
    else:
        operations += ["allreduce_1", "allreduce_2"]
    # Initialize the dictionary to log time
    T_dict_individual = {
        f'T_{operation}': np.zeros((N_timing_loop, n_layers)) 
        for operation in operations
        if operations not in ["grad_sync", "timing_loop"]  # exclude non-layer-wise op
    }
    T_dict_total = {f'T_{operation}': np.zeros(N_timing_loop) for operation in operations}
    T_grad_sync_individual = np.zeros((N_timing_loop, n_iter_grad_sync))
    TensorParallelResults = namedtuple(
        "TensorParallelResults",
        ["T_dict_individual", "T_dict_total", "T_grad_sync_individual", 
         "interim1", "interim2", "interim3", "interim4", "allreduce_grad"]
    )
    for i in range(N_timing_loop):
        timing_loop_start_time=sync_and_time(args.device)
        timing_loop_time = 0.0
        for l in range(n_layers):
            ## Attention Block
            if SP:
                _, T_allgather_1 = timed(lambda: dist.all_gather_into_tensor(input, partial_input, group=tp_group))
            interim1, T_QKV = timed(lambda: torch.matmul(input, attn_W_QKV.t())) # W_qkv: [s, b, h] -> [s, b, 3h]
            # TODO: FA
            interim2, T_WO = timed(lambda: torch.matmul(interim1, attn_WO.t())) # W_out: [s, b, h] -> [s, b, h]
            if SP:
                _, T_reduce_scatter_1 = timed(lambda: dist.reduce_scatter_tensor(partial_interim2, interim2, group=tp_group))
            else:
                _, T_allreduce_1 = timed(lambda: dist.all_reduce(interim2, group=tp_group))
            # Skipping dropout and Norm
            ## MLP Block
            if SP:
                _, T_allgather_2 = timed(lambda: dist.all_gather_into_tensor(interim2, partial_interim2, group=tp_group))
            interim3, T_H_4H = timed(lambda: torch.matmul(interim2, mat_h_4h.t())) # Up projection: [s, b, h] -> [s, b, 4h]
            interim4, T_4H_H = timed(lambda: torch.matmul(interim3, mat_4h_h.t())) # Down projection: [s, b, 4h] -> [s, b, h]
            if SP:
                _, T_reduce_scatter_2 = timed(lambda: dist.reduce_scatter_tensor(partial_interim4, interim4, group=tp_group))
            else:
                _, T_allreduce_2 = timed(lambda: dist.all_reduce(interim4, group=tp_group))
            # Log op's time each layer
            for timed_op, _ in T_dict_individual.items():
                if timed_op not in ["T_grad_sync", "T_timing_loop"]:
                    T_dict_individual[timed_op][i, l] = eval(timed_op)
        ## Grad sync  # TODO: make all-reduce async
        for k in range(n_iter_grad_sync):
            _, T_grad_sync_individual[i, k] = timed(lambda: dist.all_reduce(allreduce_grad, group=dp_group))
        timing_loop_end_time = sync_and_time(args.device)
        T_grad_sync = T_grad_sync_individual[i, :].sum()
        T_timing_loop = (timing_loop_end_time-timing_loop_start_time)
        timing_loop_time += T_timing_loop
        # Log op's time each iter
        # TODO: make normalization (e.g. / 1e6) consistent across all times
        for op, _ in T_dict_total.items():
            if op in ["T_grad_sync", "T_timing_loop"]:
                T_dict_total[op][i] = eval(op) / 1e6
            else:
                T_dict_total[op][i] = T_dict_individual[op][i, :].sum() / 1e6
        # logging.info(f"Input Shape = {input.shape}")
        # logging.info(f"Interim 1 Shape = {interim1.shape}")
        # logging.info(f"Interim 2 Shape = {interim2.shape}")
        # logging.info(f"Iter {m}, layer {i}, allreduce buffer = {interim2.shape}")
        # logging.info(f"Interim 3 Shape = {interim3.shape}")
        # logging.info(f"Interim 4 Shape = {interim4.shape}")
    return TensorParallelResults(
        T_dict_individual, 
        T_dict_total, 
        T_grad_sync_individual, 
        interim1, 
        interim2, 
        interim3, 
        interim4, 
        allreduce_grad
    )

def timed(function):
    '''
        lambda: func(function_to_time())
    '''
    ## TODO: Change to record GPU Events to time without synchronizing device and Host
    strt = sync_and_time(args.device)
    res = function()
    end = sync_and_time(args.device)
    return res, end-strt

# @trace_func
# def synchronize(device):
#     if device == "cuda":
#         return torch.cuda.synchronize()
#     elif device == "xpu":
#         return torch.xpu.synchronize()
#     else:
#         raise NotImplementedError("This method is not implemented yet.")
#         return None
    
@trace_func
def sync_and_time(device):
    if device not in ["cuda", "xpu"]:
        raise NotImplementedError("This method is not implemented yet.")
    elif device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()
    return time.perf_counter_ns()

@trace_func
def get_device_string(device):
    if device == "cuda": 
        return f"cuda:{torch.cuda.current_device()}" 
    elif device == "xpu":
        
        return f"xpu:{torch.xpu.current_device()}" 
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

@trace_func
def get_device_count(device):
    if device == "cuda":
        return torch.cuda.device_count() 
    elif device == "xpu":
        return torch.xpu.device_count() 
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

@trace_func
def set_device(visible_device):
    if args.device == "cuda":
        return torch.cuda.set_device(visible_device)
    elif args.device == "xpu":
        return torch.xpu.set_device(visible_device)
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

@trace_func
def get_backend(device):
    if device =="cuda":
        return "nccl"
    elif device == "xpu":
        return "ccl"
    else:
        raise NotImplementedError("This method is not implemented yet.")
        return None

def matmul_flops(input_shape, other_shape):
    """
    Calculates matmul floating point operations for torch.matmul
    """
    assert len(input_shape) > 1 #Not implemented
    assert len(other_shape) == 2 #Not implemented
    #Reference: https://math.stackexchange.com/questions/3512976/proof-of-of-flops-in-matrix-multiplication
    return np.prod(input_shape[:-1]) * other_shape[-1] * (2*other_shape[-2]-1)


warmup=args.warmup_iterations
log_directory = args.log_directory  # Directory for logs
log_filename = args.log_file  # Log file name
log_filepath = os.path.join(log_directory, log_filename)

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

logging.info(f"rank {rank}/{world_size}")

visible_device = rank % get_device_count(args.device)
local_rank = visible_device
set_device(visible_device)
#os.environ['CCL_LOCAL_RANK'] = str(device)
#os.environ['CCL_LOCAL_SIZE'] = str(device_count)
#backend = "ccl"

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

#torch.xpu.set_device(device)
dist.init_process_group(
    backend=get_backend(args.device),
    init_method="env://",
    world_size=world_size,
    rank=rank,
)

TP = args.tensor_parallel_degree
assert TP % 2 == 0

def get_tp_group(TP, world_size):
    tp_group=None
    for i in range(world_size//TP):
        ranks = [j for j in range(i*TP,(i+1)*TP)]
        if rank==0:
            logging.info(f"TP group = {ranks}")
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group=group
    return tp_group

if TP is not None:
    tp_group = get_tp_group(TP, world_size)
else:
    TP = 1
    tp_group = get_tp_group(TP, world_size)

"""
 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11
12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 """

dp_switch = args.data_parallel_switch
dp_group = None

if dp_switch:
    for i in range(TP):
        ranks = [i for i in range(i,world_size,TP)]
        if rank==0:
            logging.info(f"DP group = {ranks}")
        group = dist.new_group(ranks)
        if rank in ranks:
            dp_group=group
else:
    ranks = [i for i in range(0,world_size,TP)]
    dp_group = dist.new_group(ranks)
    if rank==0:
        logging.info(f"DP group = {ranks}")

"""
0,12,24,36
1,13,25,37
...
"""
S = args.sequence_length #4608 #4608 sequence length
H = args.hidden_dimension #9216 #9216 hidden dimension
M = 1
all_gather_buffer = torch.zeros([S, M, H], dtype=data_type, device=get_device_string(args.device))
SP=args.sequence_parallel_switch
if SP:
    assert S % TP == 0, "sequence must be dividable by TP degree with sequence parallelism enabled"

partial_input = torch.normal(
    mean=args.init_mean, 
    std=torch.ones([S//TP, M, H], dtype=data_type, device=get_device_string(args.device))*args.init_std
)
input = torch.normal(
    mean=args.init_mean, 
    std=torch.ones([S, M, H], dtype=data_type, device=get_device_string(args.device))*args.init_std
)
partial_interim2 = torch.zeros([S//TP, M, H], dtype=data_type, device=get_device_string(args.device))
partial_interim4 = torch.zeros([S//TP, M, H], dtype=data_type, device=get_device_string(args.device))

"""
if SP:
    partial_input = torch.ones([S//TP, M, H], dtype=data_type, device=get_device_string(args.device))
    input = torch.ones([S, M, H], dtype=data_type, device=get_device_string(args.device))
    partial_interim2 = torch.zeros([S//TP, M, H], dtype=data_type, device=get_device_string(args.device))
    partial_interim4 = torch.zeros([S//TP, M, H], dtype=data_type, device=get_device_string(args.device)) 
else:
    input = torch.ones([S, M, H], dtype=data_type, device=get_device_string(args.device))
"""
#logging.info(f"Input shape = {input.shape}")
attn_W_QKV = torch.normal(mean=args.init_mean, std=torch.ones(
        H//TP,
        H,
        device=get_device_string(args.device),
        dtype=data_type,
    )*args.init_std)

attn_WO = torch.normal(mean=args.init_mean, std=torch.ones(
        H,
        H//TP,
        device=get_device_string(args.device),
        dtype=data_type,
    )*args.init_std)

mat_h_4h = torch.normal(mean=args.init_mean, std=torch.ones(
        4*H//TP,
        H,
        device=get_device_string(args.device),
        dtype=data_type,
    )*args.init_std)
mat_4h_h = torch.normal(mean=args.init_mean, std=torch.ones(
        H,
        4*H//TP,
        device=get_device_string(args.device),
        dtype=data_type,
    )*args.init_std)

n_layers = args.number_of_transformer_layers
number_of_total_parameters = (
    (attn_W_QKV.shape[0]*attn_W_QKV.shape[1]
    +attn_WO.shape[0]*attn_WO.shape[1]
    +mat_h_4h.shape[0]*mat_h_4h.shape[1]
    +mat_4h_h.shape[0]*mat_4h_h.shape[1]) 
     * n_layers)
#logging.info(f"Parameters = {number_of_total_parameters / 1e9} Billions")
# number of iterations for the gradient synchronization loop

highest_bucket_size = int(args.grad_allreduce_bucket)
n_iter_grad_sync = math.ceil(number_of_total_parameters / highest_bucket_size)

allreduce_grad = torch.normal(mean=0.0, 
    std=torch.ones([highest_bucket_size], dtype=data_type, device=get_device_string(args.device))*0.01)

if rank==0:
    logging.info("start loop")

#tensor_parallel(args.warmup_iterations, args.number_of_transformer_layers, n_iter_grad_sync, warmup=True)

if args.trace is not None:
    activities=[ProfilerActivity.CPU]
    if args.device == "xpu":
        activities.append(ProfilerActivity.XPU)
    else:
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True) as prof:
        result = tensor_parallel(args.iterations, args.number_of_transformer_layers, n_iter_grad_sync, 
                                 input=input, partial_input=partial_input, partial_interim2=partial_interim2, 
                                 partial_interim4=partial_interim4, attn_W_QKV=attn_W_QKV, attn_WO=attn_WO, 
                                 mat_h_4h=mat_h_4h, 
                                 mat_4h_h=mat_4h_h, allreduce_grad=allreduce_grad)
    prof.export_chrome_trace(f"{args.log_directory}/{args.trace}-{rank}-of-{world_size}.json")
else:
    result = tensor_parallel(args.iterations, args.number_of_transformer_layers, n_iter_grad_sync, 
                             input=input, partial_input=partial_input, partial_interim2=partial_interim2, 
                             partial_interim4=partial_interim4, attn_W_QKV=attn_W_QKV, attn_WO=attn_WO,
                             mat_h_4h=mat_h_4h, mat_4h_h=mat_4h_h, allreduce_grad=allreduce_grad)

T_dict_individual = result.T_dict_individual
T_dict_total = result.T_dict_total
T_grad_sync_individual = result.T_grad_sync_individual
interim4 = result.interim4
interim3 = result.interim3
interim2 = result.interim2
interim1 = result.interim1
allreduce_grad_after = result.allreduce_grad

#tp_allreduce_data_volume = (args.sequence_length * args.hidden_dimension * data_type_multiplier)
#sp_allgather_data_volume = (args.sequence_length * data_type_multiplier)

tp_allreduce_1_data_volume = (interim2.shape[0] * interim2.shape[-1] * data_type_multiplier)
tp_allreduce_2_data_volume = (interim4.shape[0] * interim4.shape[-1] * data_type_multiplier)

sp_allgather_data_volume = ((TP - 1) * (args.sequence_length // TP) * args.hidden_dimension * data_type_multiplier) 

def format_logging_timings(text, data, key, time_multiplier=1):
    """
    TODO document here
    """
    if key is not None:
        data[key] = data[key][args.warmup_iterations:]  # Extract warmup iters
        logging.info("{text} takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
        .format(text=text, max_time=np.max(data[key])/time_multiplier, 
        min_time=np.min(data[key])/time_multiplier, avg_time=np.mean(data[key])/time_multiplier))
    else:
        data = data[args.warmup_iterations:]  # Extract warmup iters
        logging.info("{text} takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
        .format(text=text, max_time=np.max(data)/time_multiplier, 
        min_time=np.min(data)/time_multiplier, avg_time=np.mean(data)/time_multiplier))


def format_logging_flops(text, in_shape, weight_shape, layers, data, key, time_multiplier=1):
    """
    TODO document here
    """
    data[key] = data[key][args.warmup_iterations:]  # Extract warmup iters
    flops = matmul_flops(in_shape, weight_shape)*layers
    max_time=np.max(data[key])
    min_time=np.min(data[key])
    avg_time=np.mean(data[key])
    tflops_min = flops/(1e12*(max_time/time_multiplier/1000))
    tflops_max = flops/(1e12*(min_time/time_multiplier/1000))
    tflops_avg = flops/(1e12*(avg_time/time_multiplier/1000))
    logging.info("{text} TFLOPS max. {max:.4f},  min. {min:.4f}, avg. {avg:.4f}"
    .format(text=text, min=tflops_min, max=tflops_max, avg=tflops_avg))


if rank == 0:
    logging.info(f"==== Main Results ====\n")
    logging.info(f"Running with {args.precision} data type")
    logging.info(f"==== List of Arguments ====")
    logging.info(f"Sequence Length = {args.sequence_length}")
    logging.info(f"Hidden Dimension = {args.hidden_dimension}")
    logging.info(f"Number of transformer layers = {args.number_of_transformer_layers}")
    logging.info(f"Precision Type = {args.precision}")
    logging.info(f"SP Value = {SP}")
    logging.info(f"TP Degree = {TP}") 
    logging.info("==== List of Arguments ====")
    logging.info("Input mean before operations = {input_mean:4f}".format(input_mean=input.mean()))
    logging.info("Result mean after all  operations = {output_mean:4f}".format(output_mean=interim4.mean()))
    logging.info(f"Shape of the (Q,K,V) atten. matrix = {attn_W_QKV.shape}")
    logging.info(f"Shape of the WO atten. matrix = {attn_WO.shape}")
    logging.info(f"Shape of the Weight matrix (H --> 4H)= {mat_h_4h.shape}")
    logging.info(f"Shape of the Weight matrix (4H --> H)= {mat_4h_h.shape}")
    logging.info(f"Interim 2 Size = {interim2.shape}")
    logging.info(f"Interim 4 Size = {interim4.shape}")
    logging.info(f"Parameters (per rank) = {number_of_total_parameters / 1e9} Billions")
    logging.info(f"N_iter_grad_sync = {n_iter_grad_sync}")
    logging.info(f"Allgather buffer size = {(args.sequence_length * args.hidden_dimension * data_type_multiplier) / 8 / 1e6} MB")
    logging.info(f"Grad Sync Allreduce bucket size = {(highest_bucket_size * data_type_multiplier) / 8 / 1e6} MB") 
    logging.info(f"DP Allreduce Throughput = {(((highest_bucket_size * data_type_multiplier) / 8) / (T_grad_sync_individual[0,0])) * 1e3} MB/s")
    if SP:
        logging.info(f"SP Allgather data volume per layer per iteration = {(sp_allgather_data_volume ) / 8 / 1e6} MB")
        logging.info(f"SP Allgather 1 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_allgather_1'])) *  1e3} MB/s")
        logging.info(f"SP Allgather 2 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_allgather_2'])) *  1e3} MB/s")
        logging.info(f"SP Reduce-Scatter 1 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_reduce_scatter_1'])) * 1e3} MB/s")
        logging.info(f"SP Reduce-Scatter 2 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_reduce_scatter_2'])) * 1e3} MB/s")
    else:
        logging.info(f"TP Allreduce 1 data volume per layer per iteration = {(tp_allreduce_1_data_volume ) / 8 / 1e6} MB")
        logging.info(f"TP Allreduce 2 data volume per layer per iteration = {(tp_allreduce_2_data_volume ) / 8 / 1e6} MB")
        logging.info(f"TP Allreduce 1 Max. Throughput per layer per iteration = {((tp_allreduce_1_data_volume/8) / np.min(T_dict_individual['T_allreduce_1'])) * 1e3} MB/s")
        logging.info(f"TP Allreduce 2 Max. Throughput per layer per iteration = {((tp_allreduce_2_data_volume/8) / np.min(T_dict_individual['T_allreduce_2'])) * 1e3} MB/s")
    logging.info("\n==== Timings per transformer layer ====")
    if SP:
        format_logging_timings("First Allgather for SP", T_dict_individual, 'T_allgather_1', 1e6)
    format_logging_timings("Column Parallel Attention Matrix W_QKV multiplication", T_dict_individual, 'T_QKV', 1e6)
    format_logging_timings("Row Parallel Attention Matrix WO multiplication", T_dict_individual, 'T_WO', 1e6)
    if SP:
        format_logging_timings("First Reduce Scatter for SP", T_dict_individual, 'T_reduce_scatter_1', 1e6)
        format_logging_timings("Second Allgather for SP", T_dict_individual, 'T_allgather_2', 1e6)
    else:
        format_logging_timings("First Allreduce for TP", T_dict_individual, 'T_allreduce_1', 1e6)
    format_logging_timings("H --> 4H Matrix multiplication", T_dict_individual, 'T_H_4H', 1e6)
    format_logging_timings("4H --> H Matrix multiplication", T_dict_individual, 'T_4H_H', 1e6)
    if SP:
        format_logging_timings("Second Reduce Scatter for SP", T_dict_individual, 'T_reduce_scatter_2', 1e6)
    else:
        format_logging_timings("Second Allreduce for TP", T_dict_individual, 'T_allreduce_2', 1e6)
    format_logging_timings("Grad Sync Allreduce over DP groups", T_grad_sync_individual, None, 1e6)
    ###################
    logging.info("\n==== Total Times for all transformer layers ====")
    if SP:
        format_logging_timings("First Allgather for SP", T_dict_total, 'T_allgather_1')
    format_logging_timings("Column Parallel Attention Matrix W_QKV multiplication", T_dict_total, 'T_QKV')
    format_logging_timings("Row Parallel Attention Matrix WO multiplication", T_dict_total, 'T_WO')
    if SP:
        format_logging_timings("First Reduce Scatter for SP", T_dict_total, 'T_reduce_scatter_1')
        format_logging_timings("Second Allgather for SP", T_dict_total, 'T_allgather_2')
    else:
        format_logging_timings("First Allreduce for TP", T_dict_total, 'T_allreduce_1')
    format_logging_timings("H --> 4H Matrix multiplication", T_dict_total, 'T_H_4H')
    format_logging_timings("4H --> H Matrix multiplication", T_dict_total, 'T_4H_H')
    if SP:
        format_logging_timings("Second Reduce Scatter for SP", T_dict_total, 'T_reduce_scatter_2')
    else:
        format_logging_timings("Second Allreduce for TP", T_dict_total, 'T_allreduce_2')
    format_logging_timings("Grad Sync Allreduce over DP groups", T_dict_total, 'T_grad_sync')

    logging.info(f"Total time taken for {args.iterations} timing loops = {sum(T_dict_total['T_timing_loop'])} ms")
    ###################
    logging.info("\n==== TFLOPS per transformer layer ====")
    format_logging_flops("Column Parallel Attention Matrix W_QKV multiplication", input.shape, attn_W_QKV.t().shape, n_layers, T_dict_total, 'T_QKV')
    format_logging_flops("Row Parallel Attention Matrix WO multiplication", interim1.shape, attn_WO.t().shape, n_layers, T_dict_total, 'T_WO')
    format_logging_flops("H --> 4H Matrix multiplication", interim2.shape, mat_h_4h.t().shape, n_layers, T_dict_total, 'T_H_4H')
    format_logging_flops("4H --> H Matrix multiplication", interim3.shape, mat_4h_h.t().shape, n_layers, T_dict_total, 'T_4H_H')
    ###################
    logging.info("\n==== ALL RESULTS ====")
    if SP:
        logging.info(f"First allgather total times from timing loop = {T_dict_total['T_allgather_1']} ms")
        logging.info(f"First reduce scatter total times from timing loop = {T_dict_total['T_reduce_scatter_1']} ms")
    else:
        logging.info(f"First allreduce total times from timing loop = {T_dict_total['T_allreduce_1']} ms")
    logging.info(f"Attention W_QKV matrix multiplication total times from timing loop = {T_dict_total['T_QKV']} ms")
    logging.info(f"Attention WO matrix multiplication total times from timing loop = {T_dict_total['T_WO']} ms")
    logging.info(f"Weight matrix (H --> 4H) multiplication total times from timing loop = {T_dict_total['T_H_4H']} ms")
    logging.info(f"Weight matrix (4H --> H) multiplication total times from timing loop = {T_dict_total['T_4H_H']} ms")
    if SP:
        logging.info(f"Second allgather total times from timing loop = {T_dict_total['T_allgather_2']} ms")
        logging.info(f"Second reduce scatter total times from timing loop = {T_dict_total['T_reduce_scatter_2']} ms")
    else:
        logging.info(f"Second allreduce total times from timing loop = {T_dict_total['T_allreduce_2']} ms")
    logging.info(f"Grad Sync Total times from timing loop = {T_dict_total['T_grad_sync']} ms")
    logging.info(f"Timing loop times = {T_dict_total['T_timing_loop']}")
    logging.info(f"==== Finished Running ====")

if args.save:
    timing_dict = {"T_dict_individual" : result.T_dict_individual,
                    "T_dict_total": result.T_dict_total,
                    "T_grad_sync_individual" : result.T_grad_sync_individual}
    #np.save(os.path.join(result_dir, args.log_file), dict(result._asdict())) ## directly converts a namedtuple to a dictionary, doesn't play well with GPU tensors.
    result_dir = os.path.join(args.log_directory, "timings")
    result_dir = os.path.join(result_dir, args.log_file)
    print("saving to:", result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    np.save(os.path.join(result_dir, f"rank_{rank}"), timing_dict)
dist.barrier()
exit()