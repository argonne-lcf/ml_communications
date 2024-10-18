"""
A naive benchmark driver based on simple argument parsing
"""
import os
import time
import math
import argparse
import subprocess

import frameworks.pytorch.ddp.nccl.gpu_allreduce as allreduce

parser = argparse.ArgumentParser(description="parse input arguments for the gpu allreduce benchmark")

parser.add_argument("-dim", "--tensor_dimension_1d",
                        help="The size of the 1d tensor that is distributed accross the ranks per node.",
                        type=int, default=4096)
args = parser.parse_args()

allreduce.main(args.tensor_dimension_1d)

#subprocess.run(["python", "/home/hossainm/ml_communications/hardware_benchmarks/frameworks/pytorch/ddp/nccl/gpu_allreduce.py", "-dim=4096"])


