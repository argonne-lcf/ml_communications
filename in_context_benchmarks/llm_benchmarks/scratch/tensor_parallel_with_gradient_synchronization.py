#print("being import", flush=True)
from mpi4py import MPI
import os
import time
import math
import socket
import argparse
import logging

import torch
import numpy as np

#import intel_extension_for_pytorch as ipex  # noqa: F401 # type: ignore
#import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
#print("being code", flush=True)

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
parser.add_argument("-wit", "--warmup_iterations", help="number of warmup iterations",
                    type=int, default=3)

parser.add_argument("-tp_switch", "--tensor_parallel_switch", help="If TRUE, implements tensor parallelism of tp_degree",
                    type=bool, default=True)

parser.add_argument("-bucket", "--size_of_the_largest_bucket_for_grad_allreduce", help="The largest bucket of message passed to do the allreduce over the data parallel groups (in number of elements in a message).",
                    type=int, default=5e8)

parser.add_argument("-dp_switch", "--data_parallel_switch", help="If TRUE, calculates data parallelism degree based of tp_degree",
                    type=bool, default=True)

parser.add_argument("-tp_degree", "--tensor_parallel_degree", help="Tensor Parallel degree. In this context, the model is distributed across the number of (tensor parallel degree)) ranks",
                    type=int, default=None)

parser.add_argument("-sp_switch", "--sequence_parallel_switch", help="Switch sequence parallelism on or off", action='store_true')

parser.add_argument("-n_layers", "--number_of_transformer_layers", help="Number of transformer layers", type=int, default=80)

parser.add_argument("-n_timing_loops", "--number_of_timing_loops", help="Number of timing loops", type=int, default=6)

parser.add_argument("-p", "--precision", help="Data type for the elements of a tensor. float32 and bfloat16 supported.",
                    type=str, default="float32")

parser.add_argument("-dvc", "--device", help="Device type. cuda and xpu supported.",
                    type=str, default="cuda")

parser.add_argument("-f", "--log_file", help="Output file name",
                    type=str, default="sequence_parallel.log")

parser.add_argument("-dir", "--log_directory", help="Output file path",
                    type=str, default="logs/")

parser.add_argument("--logging", help="Switch logging on", action='store_true')

#parser.add_argument("-dp_degree", "--data_parallel_degree", help="Data Parallel degree. In this context, the data (tokens etc.) is distributed across the number of (data parallel degree)) ranks",
#                    type=int, default=6)

args = parser.parse_args()
warmup=args.warmup_iterations

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

print(f"rank {rank}/{world_size}")
#device_count = torch.xpu.device_count()
#device_count = int(os.environ["NGPU_PER_HOST"])

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

#d1 = 4096 #4608 sequence length
#d2 = 9216 #9216 hidden dimension
#42467328 8MB?
#600000000, 1.2GB?

#TP=12
TP = args.tensor_parallel_degree
#tp_switch = args.tensor_parallel_switch
#tp_group = None

def get_tp_group(TP, world_size):
    tp_group=None
    for i in range(world_size//TP):
        ranks = [j for j in range(i*TP,(i+1)*TP)]
        if rank==0:
            print("TP group", ranks)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            tp_group=group
    return tp_group

if TP is not None:
    tp_group = get_tp_group(TP, world_size)
else:
    TP = 1
    tp_group = get_tp_group(TP, world_size)

"""
if TP is not None:
    for i in range(world_size//TP):
        ranks = [j for j in range(i*TP,(i+1)*TP)]
        if rank==0:
            print("TP group", ranks)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            tp_group=group
else:
    TP = 1
    for i in range(world_size//TP):
        ranks = [j for j in range(i*TP,(i+1)*TP)]
        if rank==0:
            print("TP group", ranks)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            tp_group=group
    #TP = 1
    #ranks = [i for i in range(TP)]
    #tp_group = torch.distributed.new_group(ranks)
    #if rank==0:
    #    print("TP group", ranks)
"""

"""
if tp_switch:    
    if TP is not None:
        for i in range(world_size//TP):
            ranks = [j for j in range(i*TP,(i+1)*TP)]
            if rank==0:
                print("TP group", ranks)
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                tp_group=group
    else:
        TP = 1
        ranks = [i for i in range(TP)]
        tp_group = torch.distributed.new_group(ranks)
        if rank==0:
            print("TP group", ranks)
else:
    # Need to figure out the logic here. What does tp_group mean if there is no
    # tensor parallelism?    
    TP=1
"""
"""
if True:
    for i in range(world_size//TP):
        ranks = [j for j in range(i*TP,(i+1)*TP)]
        if rank==0:
            print("TP group", ranks)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            tp_group=group
else:
    ranks = [i for i in range(TP)]
    tp_group = torch.distributed.new_group(ranks)
    if rank==0:
        print("TP group", ranks)
"""

"""
 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11
12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 """

#dp_group = 2

dp_switch = args.data_parallel_switch
dp_group = None

if dp_switch:
    for i in range(TP):
        ranks = [i for i in range(i,world_size,TP)]
        if rank==0:
            print("DP group", ranks)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            dp_group=group
else:
    ranks = [i for i in range(0,world_size,TP)]
    dp_group = torch.distributed.new_group(ranks)
    if rank==0:
        print("DP group", ranks)


"""
if True:
    for i in range(TP):
        ranks = [i for i in range(i,world_size,TP)]
        if rank==0:
            print("DP group", ranks)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            dp_group=group
else:
    ranks = [i for i in range(0,world_size,TP)]
    dp_group = torch.distributed.new_group(ranks)
    if rank==0:
        print("DP group", ranks)
"""
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
    partial_input = torch.rand([S//TP, M, H], dtype=data_type, device=get_device_string(args.device))
    input = torch.zeros([S, M, H], dtype=data_type, device=get_device_string(args.device))
    partial_interim2 = torch.zeros([S//TP, M, H], dtype=data_type, device=get_device_string(args.device))
    partial_interim4 = torch.zeros([S//TP, M, H], dtype=data_type, device=get_device_string(args.device)) 
else:
    input = torch.rand([S, M, H], dtype=data_type, device=get_device_string(args.device))
## Write the if SP clause
#print(f"Input shape = {input.shape}")
mm1 = torch.rand(
        H//TP,
        H,
        device=get_device_string(args.device),
        dtype=data_type,
    )*1e-4

mm2 = torch.rand(
        H,
        H//TP,
        device=get_device_string(args.device),
        dtype=data_type,
    )*1e-3

mm3 = torch.rand(
        4*H//TP,
        H,
        device=get_device_string(args.device),
        dtype=data_type,
    )*1e-4
mm4 = torch.rand(
        H,
        4*H//TP,
        device=get_device_string(args.device),
        dtype=data_type,
    )*1e-3

n_layers = args.number_of_transformer_layers
number_of_total_parameters = ((mm1.shape[0]*mm1.shape[1] + mm2.shape[0]*mm2.shape[1] +  mm3.shape[0]*mm3.shape[1] +  mm4.shape[0]*mm4.shape[1]) * n_layers)
#print(f"Parameters = {number_of_total_parameters / 1e9} Billions")

# number of iterations for the gradient synchronization loop

highest_bucket_size = int(args.size_of_the_largest_bucket_for_grad_allreduce)
n_iter_grad_sync = math.ceil(number_of_total_parameters / highest_bucket_size)

## Number of elements on a tensor
#gather_bucket_size = 10000000
#total_bucket_size = gather_bucket_size*world_size

#allgather_grad = torch.rand([gather_bucket_size], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")
#allgather_res = torch.zeros([total_bucket_size], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")

#grad_bucket_size = 600000000
allreduce_grad = torch.rand([highest_bucket_size], dtype=data_type, device=get_device_string(args.device))


#tp_group = torch.distributed.new_group([i for i in range(12)])
#dp_group = torch.distributed.new_group([i for i in range(world_size)])

if rank==0:
    print("start loop", flush=True)
#timem1 = 0.0

def tensor_parallel(N_timing_loop, n_layers, n_iter_grad_sync):
    # Define how many variables you want
    operations = ["allgather_1", "QKV", "WO", "reduce_scatter_1",
                  "allreduce_1", "allgather_2", "H_4H", "4H_H", "reduce_scatter_2", 
                  "allreduce_2", "grad_sync", "timing_loop"]

    # Initialize the variables using a dictionary
    T_dict_individual = {f'T_{operation}': np.zeros((N_timing_loop, n_layers)) for operation in operations[:-2]}
    T_dict_total = {f'T_{operation}': np.zeros(N_timing_loop) for operation in operations}
    T_grad_sync_individual = np.zeros((N_timing_loop, n_iter_grad_sync))
 

    # Total time for the first all_gather, to collect parallelized sequences
    #time0 = 0.0
    #t_first_allgather = 0.0
    # Individual times for the first all_gather for SP in the transformer loop
    #T0 = []
    #T0 = np.zeros((N_timing_loop, n_layers))
    #list_first_allgather_times = []
    # Total times from the timing loops for the first all_gather for SP in the transformer loop
    #T0_timing_loop = np.zeros(N_timing_loop)
    # Total time for the first column parallel Attention (Q,K,V) multiplication
    #time1 = 0.0
    # Individual times for the column parallel Attention matrix multiplication
    #T1 = []
    #T1 = np.zeros((N_timing_loop, n_layers))
    # Total times from the timing loops for the first column parallel Attention matrix multiplication in the transformer loop
    #T1_timing_loop = np.zeros(N_timing_loop)
    # Total time for the row parallel Attention matrix multiplication
    #time2 = 0.0
    # Individual times for the row parallel Attention matrix multiplication
    #T2 = np.zeros((N_timing_loop, n_layers))
    # Total times from the timing loops for the first row parallel Attention matrix multiplication in the transformer loop
    #T2_timing_loop = np.zeros(N_timing_loop)
    # Total time for the first reduce scatter for SP
    #time3 = 0.0
    # Individual times for the first reduce scatter
    #T3 = []
    # Total times from the timing loops for the first reduce scatter in the transformer loop
    #T3_timing_loop=[]
    # Total time for the first Allreduce for TP
    #time4 = 0.0
    # Individual times for the first Allreduce
    #T4 = []
    # Total times from the timing loops for the first Allreduce in the transformer loop
    #T4_timing_loop=[]
    # Total time for the second allgather, after the attention layer
    #time5 = 0.0
    # Individual times for the second allgather
    #T5 = []
    # Total times from the timing loops for the second Allgather in the transformer loop
    #T5_timing_loop=[]
    # Total time for the Hidden representation H --> 4H
    #time6 = 0.0
    # Individual times for H --> 4H
    #T6 = []
    # Total times from the timing loops for the Hidden representation H --> 4H in the transformer loop
    #T6_timing_loop=[]
    # Total time for the Hidden representation 4H --> H
    #time7 = 0.0
    # Individual times for 4H --> H
    #T7 = []
    # Total times from the timing loops for the Hidden representation 4H --> H in the transformer loop
    #T7_timing_loop=[]
    # Total time for second reduce scatter
    #time8 = 0.0
    # Individual times for second reduce scatter
    #T8 = []
    # Total times from the timing loops for the second reduce scatter in the transformer loop
    #T8_timing_loop=[]
    # Total time for the second Allreduce
    #time9 = 0.0
    # Individual times for the second Allreduce
    #T9 = []
    # Total times from the timing loops for the second Allreduce in the transformer loop
    #T9_timing_loop=[]

    # Total time for grad allreduce over data groups
    #time10 = 0.0
    # Individual times for grad allreduce
    #T10 = []
    # Total times from the timing loops for the grad Allreduce
    #T10_timing_loop=[]

    #total_timing_loop_times=[]
    #timing_loop_time=0.0

    N_timing_loop = args.number_of_timing_loops ## For the full timing loop, not implemented yet
    for m in range(N_timing_loop):
        timing_loop_start_time=time.time()
        #time0 = 0.0
        t_ag_1 = 0.0
        #time1 = 0.0
        t_qkv = 0.0
        #time2 = 0.0
        t_WO = 0.0
        #time3 = 0.0
        t_rs_1 = 0.0
        #time4 = 0.0
        t_ardc_1 = 0.0
        #time5 = 0.0
        t_ag_2 = 0.0
        #time6 = 0.0
        t_h_4h = 0.0
        #time7 = 0.0
        t_4h_h = 0.0
        #time8 = 0.0
        t_rs_2 = 0.0
        #time9 = 0.0
        t_ardc_2 = 0.0
        #time10 = 0.0
        t_grad = 0.0
        timing_loop_time = 0.0
        for i in range(n_layers):
            start = time.time()
            if SP:
                torch.distributed.all_gather_into_tensor(
                    input, partial_input, group=tp_group
                )
                #torch.xpu.synchronize()
                synchronize(args.device)
                end = time.time()
                #T0.append(end-start)
                T_dict_individual["T_allgather_1"][m][i] = (end-start)
                #time0 += end - start
                t_ag_1 += end - start
            start = time.time()
            interim1 = torch.matmul(input, mm1.t())
            #torch.xpu.synchronize()
            synchronize(args.device)
            end = time.time()
            #T1.append(end - start)
            T_dict_individual["T_QKV"][m][i] = (end - start)
            #time1 += end-start
            t_qkv += end-start

            start = end
            interim2 = torch.matmul(interim1, mm2.t())
            #torch.xpu.synchronize()
            synchronize(args.device)
            end = time.time()
            #T2.append(end-start)
            T_dict_individual["T_WO"][m][i] = (end - start)
            #time2 += end-start
            t_WO += end-start

            start = time.time()
            if SP:
                torch.distributed.reduce_scatter_tensor(
                partial_interim2, interim2, group=tp_group
                )
                #torch.xpu.synchronize()
                synchronize(args.device)
                end = time.time()
                #T3.append(end-start)
                T_dict_individual["T_reduce_scatter_1"][m][i] = (end - start)
                #time3 += end-start
                t_rs_1 += end-start
            else:
                start = time.time()
                #print("Doing ALLREDUCE now")
                torch.distributed.all_reduce(
                    interim2, group=tp_group
                )
                #torch.xpu.synchronize()
                synchronize(args.device)
                end = time.time()
                #T4.append(end-start)
                T_dict_individual["T_allreduce_1"][m][i] = (end - start)
                #time4 += end-start
                t_ardc_1 += end-start
            if SP:
                start = time.time()
                torch.distributed.all_gather_into_tensor(
                    interim2, partial_interim2, group=tp_group
                )
                #torch.xpu.synchronize()
                synchronize(args.device)
                end = time.time()
                #T5.append(end-start)
                T_dict_individual["T_allgather_2"][m][i] = (end - start)
                #time5 += end-start
                t_ag_2 += end-start
            start = time.time()
            interim3 = torch.matmul(interim2, mm3.t())
            #torch.xpu.synchronize()
            synchronize(args.device)
            end = time.time()
            #T6.append(end-start)
            T_dict_individual["T_H_4H"][m][i] = (end - start)
            #time6 += end-start
            t_h_4h += end-start
            
            start = end
            interim4 = torch.matmul(interim3, mm4.t())
            #torch.xpu.synchronize()
            synchronize(args.device)
            end = time.time()
            #T7.append(end-start)
            T_dict_individual["T_4H_H"][m][i] = (end - start)
            #time7 += end-start
            t_4h_h += end-start
            #
            if SP:
                start = time.time()
                #print("Doing Reduce Scatter Now")
                torch.distributed.reduce_scatter_tensor(
                    partial_interim4, interim4, group=tp_group
                )
                #torch.xpu.synchronize()
                synchronize(args.device)
                end = time.time()
                #T8.append(end-start)
                T_dict_individual["T_reduce_scatter_2"][m][i] = (end - start)
                #time8 += end-start
                t_rs_2 += end-start
            else:
                start = time.time()
                torch.distributed.all_reduce(
                    interim4, group=tp_group
                )
                #torch.xpu.synchronize()
                synchronize(args.device)
                end = time.time()
                #T9.append(end-start)
                T_dict_individual["T_allreduce_2"][m][i] = (end - start)
                #time9 += end-start
                t_ardc_2 += end-start
        #torch.xpu.synchronize()
        synchronize(args.device)
        for k in range(n_iter_grad_sync):
            start = time.time()
            torch.distributed.all_reduce(
                allreduce_grad, group=dp_group
            )
            #torch.xpu.synchronize()
            synchronize(args.device)
            end = time.time()
            #T10.append(end-start)
            T_grad_sync_individual[m][k] = (end - start)
            #time10 += end-start
            t_grad += end-start
            #start = end
        #T10_timing_loop.append(time10 * 1000)
        #T_dict_total["T_grad_sync"][m] = (time10 * 1000)
        T_dict_total["T_grad_sync"][m] = (t_grad * 1000)
        #torch.xpu.synchronize()
        synchronize(args.device)
        #T0_timing_loop.append(time0 * 1000)
        #T0_timing_loop[m] = (time0 * 1000)
        #T_dict_total["T_allgather_1"][m] = (time0 * 1000)
        T_dict_total["T_allgather_1"][m] = (t_ag_1 * 1000)
        #T1_timing_loop[m] = (time1 * 1000)
        #T_dict_total["T_QKV"][m] = (time1 * 1000)
        T_dict_total["T_QKV"][m] = (t_qkv * 1000)
        #T2_timing_loop[m] = (time2 * 1000)
        #T_dict_total["T_WO"][m] = (time2 * 1000)
        T_dict_total["T_WO"][m] = (t_WO * 1000)
        #T3_timing_loop.append(time3 * 1000)
        #T_dict_total["T_reduce_scatter_1"][m] = (time3 * 1000)
        T_dict_total["T_reduce_scatter_1"][m] = (t_rs_1 * 1000)
        #T4_timing_loop.append(time4 * 1000)
        #T_dict_total["T_allreduce_1"][m] = (time4 * 1000)
        T_dict_total["T_allreduce_1"][m] = (t_ardc_1 * 1000)
        #T5_timing_loop.append(time5 * 1000)
        #T_dict_total["T_allgather_2"][m] = (time5 * 1000)
        T_dict_total["T_allgather_2"][m] = (t_ag_2 * 1000)
        #T6_timing_loop.append(time6 * 1000)
        #T_dict_total["T_H_4H"][m] = (time6 * 1000)
        T_dict_total["T_H_4H"][m] = (t_h_4h * 1000)
        #T7_timing_loop.append(time7 * 1000)
        #T_dict_total["T_4H_H"][m] = (time7 * 1000)
        T_dict_total["T_4H_H"][m] = (t_4h_h * 1000)
        #T8_timing_loop.append(time8 * 1000)
        #T_dict_total["T_reduce_scatter_2"][m] = (time8 * 1000)
        T_dict_total["T_reduce_scatter_2"][m] = (t_rs_2 * 1000)
        #T9_timing_loop.append(time9 * 1000)
        #T_dict_total["T_allreduce_2"][m] = (time9 * 1000)
        T_dict_total["T_allreduce_2"][m] = (t_ardc_2 * 1000)
        timing_loop_end_time=time.time()
        #total_timing_loop_times.append((timing_loop_end_time - timing_loop_start_time) * 1000)
        T_dict_total["T_timing_loop"][m] = ((timing_loop_end_time - timing_loop_start_time) * 1000)  
        timing_loop_time += (timing_loop_end_time - timing_loop_start_time)
        timing_loop_start_time = timing_loop_end_time
    return T_dict_individual, T_dict_total, T_grad_sync_individual

result = tensor_parallel(args.number_of_timing_loops, args.number_of_transformer_layers, n_iter_grad_sync)

if rank == 0:
    print(result[0]["T_allgather_1"].shape)
    print(result[0]["T_QKV"].shape)
    print(result[1]["T_allgather_1"].shape)
    print(result[1]["T_QKV"].shape)
    print(f"N_iter_grad_sync = {n_iter_grad_sync}")
    print(result[2].shape)
    print(f"First allgather total times from timing loop = {result[1]['T_allgather_1']}")
    print(f"Grad Sync Total times from timing loop = {result[1]['T_grad_sync']}")
    print(f"Timing loop times = {result[1]['T_timing_loop']}")
    print(f"SP Value = {SP}")
    print(f"TP Degree = {TP}")
    #print(f"Shape of the (Q,K,V) atten. matrix = {mm1.shape}")
    #print(f"Shape of the W_0 atten. matrix = {mm2.shape}")
    #print(f"Shape of the Weight matrix (H --> 4H)= {mm3.shape}")
    #print(f"Shape of the Weight matrix (4H --> H)= {mm4.shape}")
    #print(f"Shape of the Input after (Q,K,V) atten. matrix = {interim1.shape}")
    #print(f"Shape of the Input after W_0 atten. matrix = {interim2.shape}")
    #print(f"Shape of the Input after Weight matrix (H --> 4H)= {interim3.shape}")
    #print(f"Shape of the Input after Weight matrix (4H --> H)= {interim4.shape}")
    ###############################################
    #print(f"First Allgather for SP total time = {time0 * 1000} ms")
    #print(f"First Allgather for SP total time = {T0_timing_loop[warmup]} ms")
    #print(f"Column parallel Attention matrix multiplication total time = {time1 * 1000} ms")
    #print(f"Column parallel Attention matrix multiplication total time = {T1_timing_loop[warmup]} ms")
    #print(f"Row parallel Attention matrix multiplication total time = {time2 * 1000} ms")
    #print(f"Row parallel Attention matrix multiplication total time = {T2_timing_loop[warmup]} ms")
    #print(f"First reduce-scatter for SP total time = {time3 * 1000} ms")
    #print(f"First reduce-scatter for SP total time = {T3_timing_loop[warmup]} ms")
    #print(f"First Allreduce for TP total time = {time4 * 1000} ms")
    #print(f"First Allreduce for TP total time = {T4_timing_loop[warmup]} ms")
    #print(f"Second Allgather for SP total time = {time5 * 1000} ms")
    #print(f"Second Allgather for SP total time = {T5_timing_loop[warmup]} ms")
    #print(f"H --> 4H matrix multiplication total time = {time6 * 1000} ms")
    #print(f"4H --> H matrix multiplication total time = {time7 * 1000} ms")
    #print(f"Second reduce-scatter for SP total time = {time8 * 1000} ms")
    #print(f"Second Allreduce for TP total time = {time9 * 1000} ms")
    #print(f"Grad Sync Allreduce over DP groups total time = {time10 * 1000} ms")
    #print(f"H --> 4H matrix multiplication total time = {T6_timing_loop[warmup]} ms")
    #print(f"4H --> H matrix multiplication total time = {T7_timing_loop[warmup]} ms")
    #print(f"Second reduce-scatter for SP total time = {T8_timing_loop[warmup]} ms")
    #print(f"Second Allreduce for TP total time = {T9_timing_loop[warmup]} ms")
    #print(f"Grad Sync Allreduce over DP groups total time = {T10_timing_loop[warmup]} ms")
    ###################################################
    #print(f"Parameters = {number_of_total_parameters / 1e9} Billions")
    #print(f"Number of iterations for gradient sync allreduce = {n_iter_grad_sync}")
    #print("First Allgather total time: {} ms\n1: {} ms\n2: {} ms\n3: {} ms {} Gbit/s".format(t0*1000, t1*1000, t2*1000, t3*1000, tp3))
    #print(f"Total time taken for {N_timing_loop} timing loops = {sum(total_timing_loop_times[warmup:])} ms")
    #print(f"Individual times from the timing loop = {total_timing_loop_times[warmup:]} ms")
    #print("First Allgather for SP takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms".format(max_time=np.max(T0_timing_loop[warmup:]), 
    #min_time=np.min(T0_timing_loop[warmup:]), avg_time=np.mean(T0_timing_loop[warmup:])))
    #print("First Column Parallel Attention Matrix multiplication takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T1_timing_loop[warmup:]), 
    #min_time=np.min(T1_timing_loop[warmup:]), avg_time=np.mean(T1_timing_loop[warmup:])))
    #print("First Row Parallel Attention Matrix multiplication takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T2_timing_loop[warmup:]), 
    #min_time=np.min(T2_timing_loop[warmup:]), avg_time=np.mean(T2_timing_loop[warmup:])))
    #print("First Reduce Scatter for SP takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T3_timing_loop[warmup:]), 
    #min_time=np.min(T3_timing_loop[warmup:]), avg_time=np.mean(T3_timing_loop[warmup:])))
    #print("First Allreduce for TP takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T4_timing_loop[warmup:]), 
    #min_time=np.min(T4_timing_loop[warmup:]), avg_time=np.mean(T4_timing_loop[warmup:])))
    #print("Second Allgather for SP takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T5_timing_loop[warmup:]), 
    #min_time=np.min(T5_timing_loop[warmup:]), avg_time=np.mean(T5_timing_loop[warmup:])))
    #print("H --> 4H Matrix multiplication takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T6_timing_loop[warmup:]), 
    #min_time=np.min(T6_timing_loop[warmup:]), avg_time=np.mean(T6_timing_loop[warmup:])))
    #print("4H --> H Matrix multiplication takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T7_timing_loop[warmup:]), 
    #min_time=np.min(T7_timing_loop[warmup:]), avg_time=np.mean(T7_timing_loop[warmup:])))
    #print("Second Reduce Scatter for SP takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T8_timing_loop[warmup:]), 
    #min_time=np.min(T8_timing_loop[warmup:]), avg_time=np.mean(T8_timing_loop[warmup:])))
    #print("Second Allreduce for TP takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T9_timing_loop[warmup:]), 
    #min_time=np.min(T9_timing_loop[warmup:]), avg_time=np.mean(T9_timing_loop[warmup:])))
    #print("Grad Sync Allreduce over DP groups takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
    #.format(max_time=np.max(T10_timing_loop[warmup:]), 
    #min_time=np.min(T10_timing_loop[warmup:]), avg_time=np.mean(T10_timing_loop[warmup:])))
    #print(f"\n ===== Below we print out the full timing data after the warmups ===== \n")
    #print(f"First Allgather total times from the timing loop = {T0_timing_loop[warmup:]} ms")
    #print(f"First Column Parallel Attention Matrix multiplication total times from the timing loop = {T1_timing_loop[warmup:]} ms")
    #print(f"First Row Parallel Attention Matrix multiplication total times from the timing loop = {T2_timing_loop[warmup:]} ms")
    #print(f"First Reduce Scatter total times from the timing loop = {T3_timing_loop[warmup:]} ms")
    #print(f"First Allreduce total times from the timing loop = {T4_timing_loop[warmup:]} ms")
    #print(f"Second Allgather total times from the timing loop = {T5_timing_loop[warmup:]} ms")
    #print(f"H --> 4H Matrix multiplication total times from the timing loop = {T6_timing_loop[warmup:]} ms")
    #print(f"4H --> H Matrix multiplication total times from the timing loop = {T7_timing_loop[warmup:]} ms")
    #print(f"Second Reduce Scatter total times from the timing loop = {T8_timing_loop[warmup:]} ms")
    #print(f"Second Allreduce total times from the timing loop = {T9_timing_loop[warmup:]} ms")
    #print(f"Grad Sync Allreduce over DP groups total times from the timing loop = {T10_timing_loop[warmup:]} ms")
