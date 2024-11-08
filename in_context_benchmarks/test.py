import torch
import torch.distributed as dist
import os
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group()

## Experiments!
## torchrun --nproc-per-node 12 /lus/flare/projects/Aurora_deployment/eugene/Microbenchmark/ml_communications/in_context_benchmarks/test.py |& tee /lus/flare/projects/Aurora_deployment/eugene/Microbenchmark/ml_communications/in_context_benchmarks/test.log

############################################################# all_gather_into_tensor ################################################################
## Experiment: Does all_gather_into_tensor require the first dimension to be the gather dimension?
## Conclusion: Yes, however the regular all_gather is flexible.
# torch_out = list(torch.empty(16, 16*12, 16).tensor_split(12, dim=1))
# tensor_in = torch.randn(16, 16, 16)

# dist.all_gather(torch_out, tensor_in)

################################################################ all_to_all_single #################################################################
## Experiment: Does 2 back-to-back all2all lead to original data?
## Conclusion: Yes.
# data = torch.randn(144, 12, 12, 12)
# gathered_data1 = torch.empty(144, 12, 1, 12)
# gathered_data2 = torch.empty(12, 12, 12, 12)
# fully_gathered_data = torch.empty_like(data)

# sub_seq = data.size(0) // WORLD_SIZE
# strt_idx = RANK * sub_seq
# end_idx = strt_idx + sub_seq
# data_scattered = data[strt_idx:end_idx, :, :, :]#.unsqueeze(dim=0)

# # print(f"data_scattered: {data_scattered}")
# # print(f"data_scattered: {data_scattered.shape}")
# dist.all_to_all_single(gathered_data1, data_scattered)
# # print(f"gathered_data: {gathered_data1.shape}")
# dist.all_to_all_single(gathered_data2, gathered_data1)
# # print(f"gathered_data: {gathered_data2.shape}")
# dist.all_gather_into_tensor(fully_gathered_data, gathered_data2)
# # print(f"gathered_data: {fully_gathered_data.shape}")

# # print(f"torch.equal(data, fully_gathered_data): {torch.equal(data, fully_gathered_data)}")
# # print(f"torch.allclose(data, fully_gathered_data): {torch.allclose(data, fully_gathered_data)}")
# if RANK == 0:
#     print(f"data_scattered: {fully_gathered_data}")
#     print(f"fully_gathered_data: {fully_gathered_data}")
    

## tensor memory
    ## if the original tensor gets multiplied then the one that is sliced also gets manipulated
    # x *= 100
    # if RANK == 0:
    #     print(x[0, 0, 0, 0])
    #     print(x_SP[0, 0, 0, 0])


# does slicing (with two numbers?) maintain dimension?
# print(f"torch.tensor([1, 2, 3])[0:1]: {torch.tensor([1, 2, 3])[0:1]}")

#
test = torch.tensor([[1, 2, 3], [4, 5, 6]]).tensor_split(3, dim=-1)
print(test) ## ([[1, 4]], [[2, 5]], [[3, 6]])