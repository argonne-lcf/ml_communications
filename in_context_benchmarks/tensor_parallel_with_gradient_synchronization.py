#print("being import", flush=True)
from mpi4py import MPI
import os
import time
import math
import socket
import argparse

import torch

import intel_extension_for_pytorch as ipex  # noqa: F401 # type: ignore
import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
#print("being code", flush=True)

parser = argparse.ArgumentParser(description="parse input arguments for tensor and data  parallel partial benchmark")

parser.add_argument("-s", "--sequence_length",
                    help="Maximum sequence length. The size of the ALLGATHER buffer",
                    type=int, default=4608)
parser.add_argument("-it", "--iterations",
                    help="number of iterations for the timing loop",
                    type=int, default=18)
parser.add_argument("-wit", "--warmup_iterations", help="number of warmup iterations",
                    type=int, default=10)

parser.add_argument("-tp_switch", "--tensor_parallel_switch", help="If TRUE, implements tensor parallelism of tp_degree",
                    type=bool, default=True)

parser.add_argument("-bucket", "--size_of_the_largest_bucket_for_grad_allreduce", help="The largest bucket of message passed to do the allreduce over the data parallel groups (in number of elements in a message).",
                    type=int, default=5e8)

parser.add_argument("-dp_switch", "--data_parallel_switch", help="If TRUE, calculates data parallelism degree based of tp_degree",
                    type=bool, default=True)

parser.add_argument("-tp_degree", "--tensor_parallel_degree", help="Tensor Parallel degree. In this context, the model is distributed across the number of (tensor parallel degree)) ranks",
                    type=int, default=2)

parser.add_argument("-sp_switch", "--sequence_parallel_switch", help="Switch sequence parallelism on or off", action='store_true')

parser.add_argument("-n_layers", "--number_of_transformer_layers", help="Number of transformer layers", type=int, default=80)


#parser.add_argument("-dp_degree", "--data_parallel_degree", help="Data Parallel degree. In this context, the data (tokens etc.) is distributed across the number of (data parallel degree)) ranks",
#                    type=int, default=6)

args = parser.parse_args()

rank = int(MPI.COMM_WORLD.Get_rank())
world_size = int(MPI.COMM_WORLD.Get_size())
#print(f"rank {rank}/{world_size}")
device_count = torch.xpu.device_count()
#device_count = int(os.environ["NGPU_PER_HOST"])

device = rank % device_count
local_rank = device
os.environ['CCL_LOCAL_RANK'] = str(device)
os.environ['CCL_LOCAL_SIZE'] = str(device_count)
backend = "ccl"

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

torch.xpu.set_device(device)
torch.distributed.init_process_group(
    backend=backend,
    init_method="env://",
    world_size=world_size,
    rank=rank,
)

#d1 = 4096 #4608 sequence length
#d2 = 9216 #9216 hidden dimension
#42467328 8MB?
#600000000, 1.2GB?

#TP=12
TP = args.tensor_parallel_degree
tp_switch = args.tensor_parallel_switch
tp_group = None

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
        ranks = [i for i in range(TP)]
        tp_group = torch.distributed.new_group(ranks)
        if rank==0:
            print("TP group", ranks)
else:
    # Need to figure out the logic here. What does tp_group mean if there is no
    # tensor parallelism?    
    TP=1

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
H = 9216 #9216 hidden dimension
M = 1
all_gather_buffer = torch.zeros([S, M, H], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")
SP=args.sequence_parallel_switch

if SP:
    partial_input = torch.rand([S//TP, M, H], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")
    input = torch.zeros([S, M, H], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")
    partial_interim2 = torch.zeros([S//TP, M, H], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")
    partial_interim4 = torch.zeros([S//TP, M, H], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}") 
else:
    input = torch.rand([S, M, H], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")
## Write the if SP clause
#print(f"Input shape = {input.shape}")
mm1 = torch.rand(
        H//TP,
        H,
        device=f"xpu:{torch.xpu.current_device()}",
        dtype=torch.bfloat16,
    )*1e-4

mm2 = torch.rand(
        H,
        H//TP,
        device=f"xpu:{torch.xpu.current_device()}",
        dtype=torch.bfloat16,
    )*1e-3

mm3 = torch.rand(
        4*H//TP,
        H,
        device=f"xpu:{torch.xpu.current_device()}",
        dtype=torch.bfloat16,
    )*1e-4
mm4 = torch.rand(
        H,
        4*H//TP,
        device=f"xpu:{torch.xpu.current_device()}",
        dtype=torch.bfloat16,
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
allreduce_grad = torch.rand([highest_bucket_size], dtype=torch.bfloat16, device=f"xpu:{torch.xpu.current_device()}")


#tp_group = torch.distributed.new_group([i for i in range(12)])
#dp_group = torch.distributed.new_group([i for i in range(world_size)])

#warmup
if rank==0:
    print("warmup", flush=True)

for i in range(10):
    #torch.distributed.all_gather_into_tensor(
    #    allgather_res, allgather_grad
    #)
    torch.distributed.all_reduce(
        allreduce_grad, group=dp_group
    )
    torch.distributed.all_reduce(
        allreduce_grad, group=tp_group
    )

if rank==0:
    print("start loop", flush=True)
timem1 = 0.0
# Total time for the first all_gather, to collect parallelized sequences
time0 = 0.0
# Individual times for the first all_gather for SP in the timing loop
T0 = []
# Total time for the first column parallel Attention (Q,K,V) multiplication
time1 = 0.0
# Individual times for the column parallel Attention matrix multiplication
T1 = []
# Total time for the row parallel Attention matrix multiplication
time2 = 0.0
# Individual times for the row parallel Attention matrix multiplication
T2 = []
# Total time for the first reduce scatter for SP
time3 = 0.0
# Individual times for the first reduce scatter
T3 = []
# Total time for the first Allreduce for TP
time4 = 0.0
# Individual times for the first Allreduce
T4 = []
# Total time for the second allgather, after the attention layer
time5 = 0.0
# Individual times for the second allgather
T5 = []
# Total time for the Hidden representation H --> 4H
time6 = 0.0
# Individual times for H --> 4H
T6 = []
# Total time for the Hidden representation 4H --> H
time7 = 0.0
# Individual times for 4H --> H
T7 = []
# Total time for second reduce scatter
time8 = 0.0
# Individual times for second reduce scatter
T8 = []
# Total time for the second Allreduce
time9 = 0.0
# Individual times for the second Allreduce
T9 = []

start_time=time.time()
torch.xpu.synchronize()
N = 1 ## For the full timing loop, not implemented yet
for i in range(n_layers): 
    start = time.time()
    #torch.distributed.all_gather_into_tensor(
    #    allgather_res, allgather_grad
    #)
    if SP:
        torch.distributed.all_gather_into_tensor(
            input, partial_input, group=tp_group
        )
        #print(f"Inside the first SP call, Input shape is = {input.shape}")
        torch.xpu.synchronize()
        end = time.time()
        T0.append(end-start)
        time0 += end - start
    #print(f"Outside the first SP call, Input shape is = {input.shape}")
    start = time.time()
    interim1 = torch.matmul(input, mm1.t())
    torch.xpu.synchronize()
    end = time.time()
    T1.append(end - start)
    time1 += end-start

    start = end
    interim2 = torch.matmul(interim1, mm2.t())
    torch.xpu.synchronize()
    end = time.time()
    T2.append(end-start)
    time2 += end-start

    start = time.time()
    #start = end
    #torch.distributed.all_gather_into_tensor(
    #    allgather_res, allgather_grad
    #)
    if SP:
        torch.distributed.reduce_scatter_tensor(
        partial_interim2, interim2, group=tp_group
        )
        torch.xpu.synchronize()
        end = time.time()
        T3.append(end-start)
        time3 += end-start
    else:
        start = time.time()
        #print("Doing ALLREDUCE now")
        torch.distributed.all_reduce(
            interim2, group=tp_group
        )
        torch.xpu.synchronize()
        end = time.time()
        T4.append(end-start)
        time4 += end-start
    if SP:
        start = time.time()
        torch.distributed.all_gather_into_tensor(
            interim2, partial_interim2, group=tp_group
        )
        torch.xpu.synchronize()
        end = time.time()
        T5.append(end-start)
        time5 += end-start
    start = time.time()
    interim3 = torch.matmul(interim2, mm3.t())
    torch.xpu.synchronize()
    end = time.time()
    T6.append(end-start)
    time6 += end-start
    
    start = end
    interim4 = torch.matmul(interim3, mm4.t())
    torch.xpu.synchronize()
    end = time.time()
    T7.append(end-start)
    time7 += end-start
    #
    if SP:
        start = time.time()
        #print("Doing Reduce Scatter Now")
        torch.distributed.reduce_scatter_tensor(
            partial_interim4, interim4, group=tp_group
        )
        torch.xpu.synchronize()
        end = time.time()
        T8.append(end-start)
        time8 += end-start
    else:
        start = time.time()
        torch.distributed.all_reduce(
            interim4, group=tp_group
        )
        torch.xpu.synchronize()
        end = time.time()
        T9.append(end-start)
        time9 += end-start
    #time1 += end-start

# Total time for grad allreduce over data groups
time10 = 0.0
# Individual times for grad allreduce
T10 = []

for k in range(n_iter_grad_sync):
    start = time.time()
    torch.distributed.all_reduce(
        allreduce_grad, group=dp_group
    )
    torch.xpu.synchronize()
    end = time.time()
    T10.append(end-start)
    time10 += end-start
end_time=time.time()
torch.xpu.synchronize()
if rank == 0:
    t0 = time0/N
    t1 = time1/N
    t2 = time2/N
    t3 = time3/N
    tp3 = 0 #(total_bucket_size*16)/(1e9*t3)
    print(f"SP Value = {SP}")
    print(f"Shape of the (Q,K,V) atten. matrix = {mm1.shape}")
    print(f"Shape of the W_0 atten. matrix = {mm2.shape}")
    print(f"Shape of the Weight matrix (H --> 4H)= {mm3.shape}")
    print(f"Shape of the Weight matrix (4H --> H)= {mm4.shape}")
    print(f"Shape of the Input after (Q,K,V) atten. matrix = {interim1.shape}")
    print(f"Shape of the Input after W_0 atten. matrix = {interim2.shape}")
    print(f"Shape of the Input after Weight matrix (H --> 4H)= {interim3.shape}")
    print(f"Shape of the Input after Weight matrix (4H --> H)= {interim4.shape}")
    print(f"First Allgather for SP total time = {time0 * 1000} ms")
    print(f"Column parallel Attention matrix multiplication total time = {time1 * 1000} ms")
    print(f"Row parallel Attention matrix multiplication total time = {time2 * 1000} ms")
    print(f"First reduce-scatter for SP total time = {time3 * 1000} ms")
    print(f"First Allreduce for TP total time = {time4 * 1000} ms")
    print(f"Second Allgather for SP total time = {time5 * 1000} ms")
    print(f"H --> 4H matrix multiplication total time = {time6 * 1000} ms")
    print(f"4H --> H matrix multiplication total time = {time7 * 1000} ms")
    print(f"Second reduce-scatter for SP total time = {time8 * 1000} ms")
    print(f"Second Allreduce for TP total time = {time9 * 1000} ms")
    print(f"Grad Sync Allreduce over DP groups total time = {time10 * 1000} ms")
    print(f"Parameters = {number_of_total_parameters / 1e9} Billions")
    print(f"Number of iterations for gradient sync allreduce = {n_iter_grad_sync}")







    #print("First Allgather total time: {} ms\n1: {} ms\n2: {} ms\n3: {} ms {} Gbit/s".format(t0*1000, t1*1000, t2*1000, t3*1000, tp3))
    print("time2", (end_time-start_time)*1000)
