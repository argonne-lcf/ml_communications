import torch
from torch.profiler import record_function
import numpy as np
import argparse
import torch.distributed as dist

import time
import os
import logging

DEVICE = None

def trace_func(func):
   def wrapper(*args, **kwargs):
      ## TODO: Understand below dunder attribute(?) and see if there is a better way
      try:
         function_name = func.__func__.__qualname__
      except:
         function_name = func.__qualname__
      with record_function(function_name):
         return func(*args, **kwargs)
   return wrapper

def timed(function):
    '''
        lambda: func(function_to_time())
    '''
    ## TODO: Change to record GPU Events to time without synchronizing device and Host
    strt = sync_and_time(DEVICE)
    res = function()
    end = sync_and_time(DEVICE)
    return res, end-strt

@trace_func
def sync_and_time(device):
    if device not in ["cuda", "xpu", "cpu"]:
        raise NotImplementedError("This method is not implemented yet.")
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "xpu":
        torch.xpu.synchronize()
    return time.perf_counter_ns()

def get_device_count(device):
    '''returns the local world size'''
    if device == "cuda":
        return torch.cuda.device_count()
    elif device == "xpu":
        return torch.xpu.device_count()
    elif device == "cpu":
        return int(os.environ["LOCAL_WORLD_SIZE"])

# def set_device_and_dtype(local_rank, dtype):
## TODO: also set device such that we don't have to pass dtype or device
def set_device(local_rank):
    global DEVICE
    DEVICE = local_rank

def get_backend(device):
    if device =="cuda":
        return "nccl"
    if device == "xpu":
        return "ccl"
    if device == "cpu":
        return "gloo"
    raise NotImplementedError("This method is not implemented yet.")

def matmul_flops(input_shape, other_shape):
    """
    Calculates matmul floating point operations for torch.matmul
    """
    assert len(input_shape) > 1 #Not implemented
    assert len(other_shape) == 2 #Not implemented
    #Reference: https://math.stackexchange.com/questions/3512976/proof-of-of-flops-in-matrix-multiplication
    return np.prod(input_shape[:-1]) * other_shape[-1] * (2*other_shape[-2]-1)

def format_logging_timings(text, data, key, time_multiplier=1, warmup_iterations=1):
    """
    TODO document here
    """
    if key is not None:
        data[key] = data[key][warmup_iterations:]  # Extract warmup iters
        logging.info("{text} takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
        .format(text=text, max_time=np.max(data[key])/time_multiplier, 
        min_time=np.min(data[key])/time_multiplier, avg_time=np.mean(data[key])/time_multiplier))
    else:
        data = data[warmup_iterations:]  # Extract warmup iters
        logging.info("{text} takes max. {max_time:.4f} ms,  min. {min_time:.4f} ms, avg. {avg_time:.4f} ms"
        .format(text=text, max_time=np.max(data)/time_multiplier, 
        min_time=np.min(data)/time_multiplier, avg_time=np.mean(data)/time_multiplier))

def format_logging_flops(text, in_shape, weight_shape, layers, data, key, time_multiplier=1, warmup_iterations=1):
    """
    TODO document here
    """
    data[key] = data[key][warmup_iterations:]  # Extract warmup iters
    flops = matmul_flops(in_shape, weight_shape)*layers
    max_time=np.max(data[key])
    min_time=np.min(data[key])
    avg_time=np.mean(data[key])
    tflops_min = flops/(1e12*(max_time/time_multiplier/1000))
    tflops_max = flops/(1e12*(min_time/time_multiplier/1000))
    tflops_avg = flops/(1e12*(avg_time/time_multiplier/1000))
    logging.info("{text} TFLOPS max. {max:.4f},  min. {min:.4f}, avg. {avg:.4f}"
    .format(text=text, min=tflops_min, max=tflops_max, avg=tflops_avg))

def log_info_rank0(msg):
    assert dist.is_initialized() ## TODO
    if dist.get_rank == 0:
        logging.info(msg)

def all_gather_parameters():
    gathered_W_qkv = torch.randn(W_qkv.shape[0]*DP, W_qkv.shape[1])
    _, T_param_allgather_1 = timed(
        lambda: dist.all_gather_into_tensor(gathered_W_qkv, W_qkv, group=DP_group))
