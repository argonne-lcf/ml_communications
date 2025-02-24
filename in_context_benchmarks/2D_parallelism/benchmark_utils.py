import torch
import torch.distributed as dist
from torch.profiler import record_function
import numpy as np

import time
import logging
import contextlib


# FIXME: 1e6 for perf_ns but what about for Event.record()?
TIME_SCALE = 1e6  # nanoseconds


# def trace_func(func):
#    def wrapper(*args, **kwargs):
#       ## TODO: Understand below dunder attribute(?) and see if there is a better way
#       try:
#          function_name = func.__func__.__qualname__
#       except:
#          function_name = func.__qualname__
#       with record_function(function_name):
#          return func(*args, **kwargs)
#    return wrapper


def synchronize():
    if torch.cuda.is_available():
        return torch.cuda.synchronize()
    if torch.xpu.is_available():
        return torch.xpu.synchronize()

# @trace_func
def _time_without_blocking():
    ## FIXaME: check if perf_counter_ns have the same metric as Event.record()
    synchronize()
    return time.perf_counter_ns()  ## FIXME elapsed time is not supported by pytorch xpu 2.3
    if torch.cuda.is_available():
        event = torch.cuda.Event(enable_timing=True)
        return event
    elif torch.xpu.is_available():
        event = torch.xpu.Event(enable_timing=True)
    else:
        return time.perf_counter_ns()
    event.record()
    return event


def time_and_save_to_dict(op_name, function, dict_time):
    """
    Excute the callback function and return time-taken without overheads
    """
    strt = _time_without_blocking()
    with record_function(op_name):
        out = function()
    end = _time_without_blocking()
    has_accelerator = False  ## FIXME elapsed time is not supported by 
    # has_accelerator = (torch.cuda.is_available() or torch.xpu.is_available())
    if op_name not in dict_time:
        dict_time[op_name] = []
    if has_accelerator:
        dict_time[op_name].append(strt.elapsed_time(end))
    else:
        dict_time[op_name].append(end - strt)
    return out


def get_backend(device):
    if device not in ['cuda', 'xpu', 'cpu']:
        return NotImplementedError(f'{device} not known')
    if device == 'cuda':
        return "nccl"
    if device == 'xpu':
        return "ccl"
    return "gloo"


def matmul_flops(input_shape, other_shape):
    """
    Calculates matmul floating point operations for torch.matmul
    """
    assert len(input_shape) > 1 #Not implemented
    assert len(other_shape) == 2 #Not implemented
    #Reference: https://math.stackexchange.com/questions/3512976/proof-of-of-flops-in-matrix-multiplication
    return np.prod(input_shape[:-1]) * other_shape[-1] * (2*other_shape[-2]-1)


def get_time_statistics(op_name, lst_time, warmup_iterations=1):
    arr_unnormalized_time = np.array(lst_time[warmup_iterations:])  # Extract warmup iters
    arr_time = arr_unnormalized_time / TIME_SCALE  
    str_to_format = '{op_name} takes sum. {sum_time:.4f}ms, max. {max_time:.4f}ms, min. '\
                    '{min_time:.4f}ms, avg. {avg_time:.4f}ms'
    str_time_statistics = str_to_format.format(
        op_name=op_name, sum_time=arr_time.sum(), max_time=arr_time.max(), 
        min_time=arr_time.min(), avg_time=arr_time.mean()
    )

    return str_time_statistics


def get_flop_statistics(op_name, gemm_input_shapes, lst_time, warmup_iterations=1):
    arr_unnormalized_time = np.array(lst_time[warmup_iterations:])  # Extract warmup iters
    arr_time = arr_unnormalized_time / TIME_SCALE 
    tflop_count = matmul_flops(*gemm_input_shapes) / 1e12
    # total_flops = flops * l
    arr_tflops = tflop_count / arr_time
    str_to_format = "{text} TFLOPS max. {max:.4f},  min. {min:.4f}, avg. {avg:.4f}"
    str_flop_statistics = str_to_format.format(
        text=op_name, max=arr_tflops.max(), min=arr_tflops.min(), avg=arr_tflops.mean()
    )

    return str_flop_statistics
    

def log_info_rank0(msg, **kwargs):
    assert dist.is_initialized()
    if dist.get_rank == 0:
        logging.info(msg)


def print_rank0(msg, **kwargs):
    assert dist.is_initialized()
    if dist.get_rank == 0:
        print(msg)


def log_and_print_rank0(msg, **kwargs):
    log_info_rank0(msg)
    print_rank0(msg)


def create_new_stream():
    if torch.cuda.is_available():
        return torch.cuda.stream(torch.cuda.Stream())
    if torch.xpu.is_available():
        return torch.xpu.stream(torch.xpu.Stream())
    return contextlib.nullcontext()