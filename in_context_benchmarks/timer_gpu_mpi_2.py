from mpi4py import MPI

import torch
import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = None
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def start(self):
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)
        self.start_time.record()

    def stop(self):
        self.end_time.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start_time.elapsed_time(self.end_time)
        if self.rank == 0:
            all_times = self.comm.gather(self.elapsed_time, root=0)
        else:
            self.comm.gather(self.elapsed_time, root=0)

        if self.rank == 0:
            self.elapsed_time = max(all_times)
            print(f"Max timing = {self.elapsed_time} ms")

    def get_time(self):
        return self.elapsed_time / 1000.0  # Convert to milliseconds


# Usage Example
timer = Timer()
tensor = torch.randn(1024, 1024, device='cuda')

timer.start()
#result = tensor.mm(tensor)
result = torch.matmul(tensor, tensor)
timer.stop()

print(f"Matrix multiplication took {timer.get_time():.4f} ms")
