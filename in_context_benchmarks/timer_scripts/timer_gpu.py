import torch

class GPUTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0.0

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        # Waits for everything to finish running on the GPU
        torch.cuda.synchronize()
        elapsed = self.start_event.elapsed_time(self.end_event)  # Time in milliseconds
        self.elapsed_time += elapsed
        return elapsed / 1000.0  # Convert milliseconds to seconds

    def get_duration(self):
        return self.elapsed_time / 1000.0  # Convert total milliseconds to seconds

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

print(f"Elapsed GPU time: {elapsed:.6f} seconds")
