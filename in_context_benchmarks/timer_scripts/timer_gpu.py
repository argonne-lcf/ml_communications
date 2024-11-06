from mpi4py import MPI
import torch

class GPUTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0.0
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        # Waits for everything to finish running on the GPU
        torch.cuda.synchronize()
        #torch.cuda.current_stream().wait_event(self.end_event)
        #self.end_event.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event)  # Time in milliseconds
        self.elapsed_time += elapsed
        if self.rank == 0:
            all_times = self.comm.gather(self.elapsed_time, root=0)
        else:
            self.comm.gather(self.elapsed_time, root=0)

        if self.rank == 0:
            self.elapsed_time = max(all_times)
            print(f"Max timing = {self.elapsed_time} milliseconds")
        return elapsed

    def get_duration(self):
        return self.elapsed_time

# Usage example
gpu_timer = GPUTimer()

x = torch.rand(1000, 1000, device="cuda")  # Allocate tensor on GPU

# Example to time a PyTorch GPU operation
def some_gpu_pytorch_operation(x):
    y = torch.matmul(x, x)
    return y

# Time the operation
gpu_timer.start()
result = some_gpu_pytorch_operation(x)
elapsed = gpu_timer.stop()

print(f"Elapsed GPU time: {elapsed:.6f} milliseconds")
