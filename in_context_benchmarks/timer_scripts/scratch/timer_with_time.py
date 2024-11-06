from mpi4py import MPI

import time
import torch

class Timer:
    def __init__(self):
        self.start_time = None
        self.duration = 0

    def start(self):
        if self.start_time is not None:
            raise RuntimeError("Timer is already running. Use stop() to stop it before starting it again.")
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer is not running. Use start() to start it.")
        torch.cuda.synchronize()
        elapsed_time = time.time() - self.start_time
        self.duration += elapsed_time
        self.start_time = None
        return elapsed_time

    def get_duration(self):
        """Returns the total duration the timer has run in seconds."""
        return self.duration

# Usage
timer = Timer()

# Example to time a PyTorch operation
def some_pytorch_operation():
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    return y

# Time the operation
timer.start()
result = some_pytorch_operation()
elapsed = timer.stop()

print(f"Elapsed time: {elapsed:.6f} seconds")
