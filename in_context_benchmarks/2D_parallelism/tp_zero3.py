# from mpi4py import MPI
from benchmark_utils import (
    timed, sync_and_time, get_device_count, set_device, get_backend, 
    matmul_flops, format_logging_timings, format_logging_flops, log_info_rank0)

import torch
import numpy as np
import torch.distributed
import torch.distributed.device_mesh
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist
import torch.nn.functional as F

import os
import time
import math
# import socket
import argparse
import logging
# from collections import namedtuple

parser = argparse.ArgumentParser(description="parse input arguments for tensor and data  parallel partial benchmark")

## TODO: Add group arguments such as model related hidden_dim... etc.
parser.add_argument("-s", "--sequence_length", type=int, default=4608,
                    help="Maximum sequence length. The size of the ALLGATHER buffer")
parser.add_argument("-d", "--hidden_dimension", type=int, default=9216,
                    help='Hidden dimension for the matrix multiplication. Proxy for the' 'model size.')
parser.add_argument("-hc", "--head-count", type=int, default=32,
                    help='number of head count for multi-headed-attention.')
parser.add_argument("-it", "--iterations", type=int, default=3,
                    help="number of iterations for the timing loop")
parser.add_argument("-wit", "--warmup_iterations", type=int, default=2,
                    help="number of warmup iterations")

parser.add_argument("-n_layers", "--number_of_transformer_layers", type=int, default=80,
                    help="Number of transformer layers")
parser.add_argument("--init_std", type=float, default=0.01,
                    help='Standard deviation for initializing weights and'                'inputs with normal distribution')
parser.add_argument("--micro-batch-size", type=int, default=1,
                    help='local batch size per model instance')
# parser.add_argument("--init_mean", type=float, default=0.0,
#                     help='Mean for initializing weights and inputs with normal' 'distribution')
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
parser.add_argument('--save', action='store_true', 
                    help='Save detail results in npy format. Generates huge files, use' 'with caution') ## 
parser.add_argument("--trace", action='store_true', help='use pytorch profiler to trace')
parser.add_argument("-sp_switch", "--sequence-parallel-switch", action='store_true',
                    help="Switch sequence parallelism on or off")

parser.add_argument("-bucket", "--grad_allreduce_bucket", type=int, default=5e8,
                    help='The largest bucket of message passed to do the allreduce over' 
                    'the data parallel groups (in number of elements in a message).')

parser.add_argument("-dp_switch", "--data_parallel_switch", action='store_true',
                    help="If TRUE, calculates data parallelism degree based of tp_degree")
parser.add_argument("-TP", "--tensor_parallel_degree", type=int, default=1,
                    help='Tensor Parallel degree. In this context, the model is'
                    'distributed across the number of (tensor parallel degree)) ranks')
parser.add_argument("--zero3", action='store_true', help='enable parameter partioning'
                    'across DP')

## TODO: Use better wording
# parser.add_argument('--inner_most_parallelism', choices={'TP', 'ulysses'}, type=str,
#                     default='ulysses', help='Which model parallelism to use as intra-node or be the first parallelism dimension')
parser.add_argument('--use-zero3', action='store_true',
                    help='Asynchronously all-gather parameters of next layer to reduce memory.'
                    'Next layer kernel will wait until the all-gather is completed')
parser.add_argument('--include-flash-attention', action='store_true', 
                    help='time and benchmark flash attention kernel by using '
                    'F.scaled_dot_product_attention')

## TODO: Why is DP switch an option? 
args = parser.parse_args()

def main():
    # warmup=args.warmup_iterations
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
    # rank = int(MPI.COMM_WORLD.Get_rank())
    # world_size = int(MPI.COMM_WORLD.Get_size())

    ## torchrun
    # os.environ["MASTER_ADDR"]="localhost"
    # os.environ["USE_LIBUV"] = "0"
        # init_method="env://?use_libuv=False"
    dist.init_process_group(backend=get_backend(args.device))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.device == 'cpu':
        local_rank = 'cpu'
    else:
        local_rank = os.environ["LOCAL_RANK"]

    ## TODO: Make the following more elegant.
    set_device(args.device)
    torch.set_default_device(args.device)
    torch.set_default_dtype(data_type)

    if rank == 0:
        print(f"args: {args}")
        print(f"world_size: {world_size}")
    print(f"rank: {rank}")
    print(f"local_rank: {local_rank}")
    if args.logging:
        if rank == 0 and not os.path.exists(log_directory):
            # Create log directory if it doesn't exist
            os.makedirs(log_directory)
        dist.barrier()
        logging.basicConfig(filename=log_filepath, filemode="a", level="INFO")
    else:
        logging.basicConfig(level="INFO")
    logging.info(f"rank {rank}/{world_size}")

    ## Initialize communication group
    assert args.head_count % args.tensor_parallel_degree == 0
    assert world_size % args.tensor_parallel_degree == 0
    data_parallel_degree = world_size // args.tensor_parallel_degree
    device_mesh_shape = [args.tensor_parallel_degree, data_parallel_degree]
    mesh_dim_names = ['TP', 'DP']
    device_mesh = torch.distributed.device_mesh.init_device_mesh(
        device_type="cpu", mesh_shape=device_mesh_shape, mesh_dim_names=mesh_dim_names
    )

    ## Initialize output buffers and weights
    S = args.sequence_length #4608 #4608 sequence length
    H = args.hidden_dimension #9216 #9216 hidden dimension
    B = args.micro_batch_size
    TP = args.tensor_parallel_degree
    DP = data_parallel_degree
    qkv = 3
    # partition sequence if sequence parallelism is enabled
    if args.sequence_parallel_switch:
        assert S % TP == 0, \
            'sequence length must be dividable by TP degree when sequence parallelism' \
            'is enabled'
        partial_x = torch.randn(S//TP, B, H)
    else:
        partial_x = torch.randn(S, B, H)
    print(f"partial_x.shape: {partial_x.shape}")
    gathered_x = torch.empty([S, B, H], dtype=data_type, device=local_rank)
    if args.use_zero3:
        sharded_hidden_dim = H//TP//DP
    else:
        sharded_hidden_dim = H//TP
    W_qkv = torch.randn(qkv*sharded_hidden_dim, H) * 0.01
    W_o = torch.randn(H, sharded_hidden_dim) * 0.01
    W_h_4h = torch.randn(4*sharded_hidden_dim, H) * 0.01
    W_4h_h = torch.randn(H, 4*sharded_hidden_dim) * 0.01

    num_total_parameters = (
        W_qkv.numel() + W_o.numel() + W_h_4h.numel() + W_4h_h.numel()
    ) * args.number_of_transformer_layers
    log_info_rank0(f"Parameters = {num_total_parameters / 1e9} Billions")
    highest_bucket_size = int(args.grad_allreduce_bucket)
    # number of iterations for the gradient synchronization loop
    n_iter_grad_sync = math.ceil(num_total_parameters / highest_bucket_size)
    if highest_bucket_size < num_total_parameters:
        highest_bucket_size = num_total_parameters
        log_info_rank0('Bucket size is due too big compared to num parameters.'
                       'Adjusting bucket size to the max num parameters')
    allreduce_grad = torch.randn(highest_bucket_size, dtype=data_type, device=local_rank)

    ## Start profiling if enabled
    log_info_rank0("start loop")
    if args.trace:
        activities=[ProfilerActivity.CPU]
        if args.device == "xpu":
            activities.append(ProfilerActivity.XPU)
        else:
            activities.append(ProfilerActivity.CUDA)
        prof = profile(activities=activities, record_shapes=True)
        prof.start()
    else:
        prof = None

    ## Create a dictionary to log time
    N_timing_loop = args.iterations
    n_layers = args.number_of_transformer_layers
    layer_operation = ["QKV", "WO", "H_4H", "4H_H"]
    if args.sequence_parallel_switch:
        layer_operation += ["allgather_1", "allgather_2", "reduce_scatter_1", 
                            "reduce_scatter_2"]
    else:
        layer_operation += ["allreduce_1", "allreduce_2"]
    if args.use_zero3:
        layer_operation += ['W_qkv_allgather', 'W_o_allgather', 'W_h_4h_allgather', 
                            'W_4h_h_allgather']
    if args.include_flash_attention:
        layer_operation.append('flash_attention')
    all_operations = layer_operation + ["grad_sync", "timing_loop"]
    T_dict_individual = {f'T_{operation}': np.zeros((N_timing_loop, n_layers)) 
                         for operation in layer_operation}
    T_dict_total = {f'T_{operation}': np.zeros(N_timing_loop) 
                    for operation in all_operations}
    T_grad_sync_individual = np.zeros((N_timing_loop, n_iter_grad_sync))
    # results = namedtuple(
    #     "TensorParallelResults",
    #     ["T_dict_individual", "T_dict_total", "T_grad_sync_individual", 
    #      "interim1", "interim2", "interim3", "interim4", "allreduce_grad"]
    # )

    ## Run and time pseudo parallel transformer
    ## TODO: set up alias for each arguments instead of spreaded out through out?
    hc = args.head_count
    assert H % hc == 0
    assert hc % TP == 0
    hs = H // hc
    TP_group = device_mesh['TP'].get_group()
    DP_group = device_mesh['DP'].get_group()
    for i in range(N_timing_loop):
        timing_loop_start_time = sync_and_time(args.device)
        timing_loop_time = 0.0
        for l in range(n_layers):
            ## TODO: Unnest by abstracting out single Transformer Layer
            # Pre-Attention (W_qkv) -> [s, b, 3h]
            if args.use_zero3:
                sharded_W_qkv = W_qkv
                W_qkv = torch.empty(sharded_W_qkv.shape[0]*DP, sharded_W_qkv.shape[1])
                _, T_W_qkv_allgather = timed(lambda: dist.all_gather_into_tensor(
                    W_qkv, sharded_W_qkv, group=DP_group, async_op=True)
                )
            if args.sequence_parallel_switch:
                _, T_allgather_1 = timed(
                    lambda: dist.all_gather_into_tensor(gathered_x, partial_x, group=TP_group)
                )
            ## TODO: do zero3 comm on a separate stream and time correctly
            gathered_x, T_QKV = timed(lambda: torch.matmul(gathered_x, W_qkv.t()))

            # Flash Attention -> (s, b, h//TP)
            if args.use_zero3:
                sharded_W_o = W_o
                W_o = torch.empty(sharded_W_o.shape[0]*DP, sharded_W_o.shape[1])
                _, T_W_o_allgather = timed(lambda: dist.all_gather_into_tensor(
                    W_o, sharded_W_o, group=DP_group, async_op=True)
                )
            q, k, v = gathered_x.view(S, B, hc//TP, qkv, hs).unbind(-2)
            if args.include_flash_attention:
                att_out, T_flash_attention = timed(
                    lambda: F.scaled_dot_product_attention(q, k, v)
                )
            else:
                att_out = q
            att_out = att_out.permute(0, 2, 1, 3).reshape(S, B, H//TP)

            # W_out -> [s, b, h]
            gathered_x, T_WO = timed(lambda: torch.matmul(att_out, W_o.t())) 
            if args.use_zero3:
                sharded_W_h_4h = W_h_4h
                W_h_4h = torch.empty(sharded_W_h_4h.shape[0]*DP, sharded_W_h_4h.shape[1])
                _, T_W_h_4h_allgather = timed(lambda: dist.all_gather_into_tensor(
                    W_h_4h, sharded_W_h_4h, group=DP_group, async_op=True)
                )
            if args.sequence_parallel_switch:
                _, T_reduce_scatter_1 = timed(
                    lambda: dist.reduce_scatter_tensor(partial_x, gathered_x, group=TP_group))
            else:
                _, T_allreduce_1 = timed(lambda: dist.all_reduce(gathered_x, group=TP_group))
            
            # Skipping dropout and Norm
            
            # MLP (Up projection) -> [s, b, 4h]
            if args.sequence_parallel_switch:
                _, T_allgather_2 = timed(
                    lambda: dist.all_gather_into_tensor(gathered_x, partial_x, group=TP_group))
            partial_proj, T_H_4H = timed(lambda: torch.matmul(gathered_x, W_h_4h.t())) 
            if args.use_zero3:
                sharded_W_4h_h = W_4h_h
                W_4h_h = torch.empty(sharded_W_4h_h.shape[0]*DP, sharded_W_4h_h.shape[1])
                _, T_W_4h_h_allgather = timed(lambda: dist.all_gather_into_tensor(
                    W_4h_h, sharded_W_4h_h, group=DP_group, async_op=True))

            # MLP (Down projection) -> [s, b, h]
            gathered_x, T_4H_H = timed(lambda: torch.matmul(partial_proj, W_4h_h.t()))
            if args.sequence_parallel_switch:
                _, T_reduce_scatter_2 = timed(
                    lambda: dist.reduce_scatter_tensor(partial_x, gathered_x, group=TP_group))
            else:
                _, T_allreduce_2 = timed(lambda: dist.all_reduce(gathered_x, group=TP_group))

            # FIXME: Remove debugging print
            if dist.get_rank() == 0:
                print(f"{i} loop at layer {l}")

            # Log op's time each layer
            for timed_op, _ in T_dict_individual.items():
                T_dict_individual[timed_op][i, l] = eval(timed_op)

        # Grad sync  # TODO: make all-reduce async
        for k in range(n_iter_grad_sync):
            _, T_grad_sync_individual[i, k] = timed(lambda: dist.all_reduce(allreduce_grad, group=TP_group))
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

        if args.trace:
            prof.step()

    if args.trace:
        prof.stop()
        prof.export_chrome_trace(f"{args.log_directory}/{args.trace}-{rank}-of-{world_size}.json")

    tp_allreduce_data_volume = S * B * H * data_type_multiplier
    sp_allgather_data_volume = (args.sequence_length // TP * args.hidden_dimension * data_type_multiplier)

    if rank == 0:
        logging.info(f"==== Main Results ====\n")
        logging.info(f"Running with {args.precision} data type")
        logging.info(f"==== List of Arguments ====")
        logging.info(f"Sequence Length = {args.sequence_length}")
        logging.info(f"Hidden Dimension = {args.hidden_dimension}")
        logging.info(f"Number of transformer layers = {args.number_of_transformer_layers}")
        logging.info(f"Precision Type = {args.precision}")
        logging.info(f"SP Value = {args.sequence_parallel_switch}")
        logging.info(f"TP Degree = {TP}") 
        logging.info("==== List of Arguments ====")
        # logging.info("Input mean before operations = {input_mean:4f}".format(input_mean=partial_x.mean()))
        # logging.info("Result mean after all  operations = {output_mean:4f}".format(output_mean=interim4.mean()))
        logging.info(f"Shape of the (Q,K,V) atten. matrix = {W_qkv.shape}")
        logging.info(f"Shape of the WO atten. matrix = {W_o.shape}")
        logging.info(f"Shape of the Weight matrix (H --> 4H)= {W_h_4h.shape}")
        logging.info(f"Shape of the Weight matrix (4H --> H)= {W_4h_h.shape}")
        # logging.info(f"Interim 2 Size = {interim2.shape}")
        # logging.info(f"Interim 4 Size = {interim4.shape}")
        logging.info(f"Parameters (per rank) = {num_total_parameters / 1e9} Billions")
        logging.info(f"N_iter_grad_sync = {n_iter_grad_sync}")
        logging.info(f"Allgather buffer size = {(args.sequence_length * args.hidden_dimension * data_type_multiplier) / 8 / 1e6} MB")
        logging.info(f"Grad Sync Allreduce bucket size = {(highest_bucket_size * data_type_multiplier) / 8 / 1e6} MB") 
        logging.info(f"DP Allreduce Throughput = {(((highest_bucket_size * data_type_multiplier) / 8) / (T_grad_sync_individual[0,0])) * 1e3} MB/s")
        if args.sequence_parallel_switch:
            logging.info(f"SP Allgather data volume per layer per iteration = {(sp_allgather_data_volume ) / 8 / 1e6} MB")
            logging.info(f"SP Allgather 1 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_allgather_1'])) *  1e3} MB/s")
            logging.info(f"SP Allgather 2 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_allgather_2'])) *  1e3} MB/s")
            logging.info(f"SP Reduce-Scatter 1 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_reduce_scatter_1'])) * 1e3} MB/s")
            logging.info(f"SP Reduce-Scatter 2 Max. Throughput = {((sp_allgather_data_volume / 8) / np.min(T_dict_individual['T_reduce_scatter_2'])) * 1e3} MB/s")
        else:
            logging.info(f"TP Allreduce 1 data volume per layer per iteration = {(tp_allreduce_data_volume ) / 8 / 1e6} MB")
            # logging.info(f"TP Allreduce 2 data volume per layer per iteration = {(tp_allreduce_2_data_volume ) / 8 / 1e6} MB")
            logging.info(f"TP Allreduce 1 Max. Throughput per layer per iteration = {((tp_allreduce_data_volume/8) / np.min(T_dict_individual['T_allreduce_1'])) * 1e3} MB/s")
            # logging.info(f"TP Allreduce 2 Max. Throughput per layer per iteration = {((tp_allreduce_2_data_volume/8) / np.min(T_dict_individual['T_allreduce_2'])) * 1e3} MB/s")
        logging.info("\n==== Timings per transformer layer ====")
        if args.tensor_parallel_degree > 1 and args.sequence_parallel_switch:
            format_logging_timings("First Allgather for SP", T_dict_individual, 'T_allgather_1', 1e6, args.warmup_iterations)
            format_logging_timings("First Reduce Scatter for SP", T_dict_individual, 'T_reduce_scatter_1', 1e6, args.warmup_iterations)
            format_logging_timings("Second Allgather for SP", T_dict_individual, 'T_allgather_2', 1e6, args.warmup_iterations)
            format_logging_timings("Second Reduce Scatter for SP", T_dict_individual, 'T_reduce_scatter_2', 1e6, args.warmup_iterations)
            format_logging_timings("First Allgather for SP", T_dict_total, 'T_allgather_1', args.warmup_iterations)
            format_logging_timings("Second Reduce Scatter for SP", T_dict_total, 'T_reduce_scatter_2', args.warmup_iterations)
            format_logging_timings("First Reduce Scatter for SP", T_dict_total, 'T_reduce_scatter_1', args.warmup_iterations)
            format_logging_timings("Second Allgather for SP", T_dict_total, 'T_allgather_2', args.warmup_iterations)
            logging.info(f"Second allgather total times from timing loop = {T_dict_total['T_allgather_2']} ms")
            logging.info(f"Second reduce scatter total times from timing loop = {T_dict_total['T_reduce_scatter_2']} ms")
            logging.info(f"First allgather total times from timing loop = {T_dict_total['T_allgather_1']} ms")
            logging.info(f"First reduce scatter total times from timing loop = {T_dict_total['T_reduce_scatter_1']} ms")
        elif args.tensor_parallel_degree > 1:
            format_logging_timings("First Allreduce for TP", T_dict_individual, 'T_allreduce_1', 1e6, args.warmup_iterations)
            format_logging_timings("Second Allreduce for TP", T_dict_individual, 'T_allreduce_2', 1e6, args.warmup_iterations)
            format_logging_timings("First Allreduce for TP", T_dict_total, 'T_allreduce_1', args.warmup_iterations)
            format_logging_timings("Second Allreduce for TP", T_dict_total, 'T_allreduce_2', args.warmup_iterations)
            logging.info(f"First allreduce total times from timing loop = {T_dict_total['T_allreduce_1']} ms")
            logging.info(f"Second allreduce total times from timing loop = {T_dict_total['T_allreduce_2']} ms")
        if args.use_zero3:
            format_logging_timings("W_qkv_allgather", T_dict_individual, 'T_W_qkv_allgather', 1e6, args.warmup_iterations)
            format_logging_timings("W_o_allgather", T_dict_individual, 'T_W_o_allgather', 1e6, args.warmup_iterations)
            format_logging_timings("W_h_4h_allgather", T_dict_total, 'T_W_h_4h_allgather', 1e6, args.warmup_iterations)
            format_logging_timings("W_4h_h_allgather", T_dict_total, 'T_W_4h_h_allgather', 1e6, args.warmup_iterations)
        if args.include_flash_attention:
            format_logging_timings('Flash Attention', T_dict_total, 'T_flash_attention', args.warmup_iterations)
        format_logging_timings("Column Parallel Attention Matrix W_QKV multiplication", T_dict_individual, 'T_QKV', 1e6, args.warmup_iterations)
        format_logging_timings("Row Parallel Attention Matrix WO multiplication", T_dict_individual, 'T_WO', 1e6, args.warmup_iterations)
        format_logging_timings("H --> 4H Matrix multiplication", T_dict_individual, 'T_H_4H', 1e6, args.warmup_iterations)
        format_logging_timings("4H --> H Matrix multiplication", T_dict_individual, 'T_4H_H', 1e6, args.warmup_iterations)
        format_logging_timings("Grad Sync Allreduce over DP groups", T_grad_sync_individual, None, 1e6, args.warmup_iterations)
        ###################
        logging.info("\n==== Total Times for all transformer layers ====")
        format_logging_timings("Column Parallel Attention Matrix W_QKV multiplication", T_dict_total, 'T_QKV', args.warmup_iterations)
        format_logging_timings("Row Parallel Attention Matrix WO multiplication", T_dict_total, 'T_WO', args.warmup_iterations)
        format_logging_timings("H --> 4H Matrix multiplication", T_dict_total, 'T_H_4H', args.warmup_iterations)
        format_logging_timings("4H --> H Matrix multiplication", T_dict_total, 'T_4H_H', args.warmup_iterations)
        format_logging_timings("Grad Sync Allreduce over DP groups", T_dict_total, 'T_grad_sync', args.warmup_iterations)
        logging.info(f"Total time taken for {args.iterations} timing loops = {sum(T_dict_total['T_timing_loop'])} ms")
        ###################
        logging.info("\n==== TFLOPS per transformer layer ====")
        format_logging_flops("Column Parallel Attention Matrix W_QKV multiplication", partial_x.shape, W_qkv.t().shape, n_layers, T_dict_total, 'T_QKV')
        # format_logging_flops("Row Parallel Attention Matrix WO multiplication", interim1.shape, W_o.t().shape, n_layers, T_dict_total, 'T_WO')
        # format_logging_flops("H --> 4H Matrix multiplication", interim2.shape, W_h_4h.t().shape, n_layers, T_dict_total, 'T_H_4H')
        # format_logging_flops("4H --> H Matrix multiplication", interim3.shape, W_4h_h.t().shape, n_layers, T_dict_total, 'T_4H_H')
        ###################
        logging.info("\n==== ALL RESULTS ====")
        logging.info(f"Attention W_QKV matrix multiplication total times from timing loop = {T_dict_total['T_QKV']} ms")
        logging.info(f"Attention WO matrix multiplication total times from timing loop = {T_dict_total['T_WO']} ms")
        logging.info(f"Weight matrix (H --> 4H) multiplication total times from timing loop = {T_dict_total['T_H_4H']} ms")
        logging.info(f"Weight matrix (4H --> H) multiplication total times from timing loop = {T_dict_total['T_4H_H']} ms")
        logging.info(f"Grad Sync Total times from timing loop = {T_dict_total['T_grad_sync']} ms")
        logging.info(f"Timing loop times = {T_dict_total['T_timing_loop']}")
        logging.info(f"==== Finished Running ====")

    if args.save:
        timing_dict = {"T_dict_individual" : T_dict_individual,
                        "T_dict_total": T_dict_total,
                        "T_grad_sync_individual" : T_grad_sync_individual}
        #np.save(os.path.join(result_dir, args.log_file), dict(result._asdict())) ## directly converts a namedtuple to a dictionary, doesn't play well with GPU tensors.
        result_dir = os.path.join(args.log_directory, "timings")
        result_dir = os.path.join(result_dir, args.log_file)
        print("saving to:", result_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        np.save(os.path.join(result_dir, f"rank_{rank}"), timing_dict)
    dist.barrier()
    exit()

if __name__ == "__main__":
    main()