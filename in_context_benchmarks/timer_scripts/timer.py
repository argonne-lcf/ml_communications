from mpi4py import MPI

import time
import torch

class Timer:
    def __init__(self):
        self.start_time = None
        self.duration = 0.0
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("Timer is already running. Use stop() to stop it before starting it again.")
        self.start_time = time.perf_counter_ns()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not running. Use start() to start it.")
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter_ns() - self.start_time
        self.duration += elapsed_time
        self.start_time = None
        if self.rank == 0:
            all_times = self.comm.gather(self.duration, root=0)
        else:
            self.comm.gather(self.duration, root=0)

        if self.rank == 0:
            self.duration = max(all_times)
            print(f"Max timing = {self.duration / 1e6} milliseconds")
        return elapsed_time / 1e6  # Convert nanoseconds to milliseconds

    def get_duration(self):
        """Returns the total duration the timer has run in seconds."""
        return self.duration / 1e6  # Convert nanoseconds to milliseconds

# Usage
'''
timer = Timer()
x = torch.rand(1000, 1000, device="cuda")

# Example to time a PyTorch operation
def some_pytorch_operation(x):
    y = torch.matmul(x, x)
    return y

# Time the operation
timer.start()
result = some_pytorch_operation(x)
elapsed = timer.stop()

print(f"Elapsed time: {elapsed:.6f} milliseconds")
'''
