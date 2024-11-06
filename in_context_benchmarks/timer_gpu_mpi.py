from mpi4py import MPI
import torch

class GPUTimer:
    def __init__(self, comm=MPI.COMM_WORLD):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0.0
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def start(self):
        #self.comm.Barrier()  # Synchronize all processes
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event)  # Time in milliseconds
        self.elapsed_time += elapsed / 1000.0  # Convert milliseconds to seconds
        return self.elapsed_time

    def get_duration(self):
        total_duration = self.comm.reduce(self.elapsed_time, op=MPI.SUM, root=0)
        if self.rank == 0:
            return total_duration / self.size
        return None

# Usage example
gpu_timer = GPUTimer()

# Example to time a PyTorch GPU operation
def some_gpu_pytorch_operation():
    x = torch.rand(1000, 1000).cuda()  # Allocate tensor on GPU
    y = torch.matmul(x, x)
    return y

# Time the operation
gpu_timer.start()
result = some_gpu_pytorch_operation()
elapsed = gpu_timer.stop()

if gpu_timer.rank == 0:
    print(f"Elapsed GPU time on average across processes: {gpu_timer.get_duration():.6f} seconds")
