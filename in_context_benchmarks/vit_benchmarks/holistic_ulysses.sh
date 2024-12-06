#!/bin/bash +x

SCRIPT_PTH=/eagle/datascience/eku/ml_communications/in_context_benchmarks/vit_benchmarks/qsub_ulysses.sh ## use abs_path
SCRIPT_DIR=$(dirname $SCRIPT_PTH)

module load conda
conda activate base

master_node=$(head -1 $PBS_NODEFILE)
## TODO: make below agnostic to aurora, sunspot, polaris
ngpus_per_node=4
# ngpus_per_node=12
num_nodes=$(wc -l < $PBS_NODEFILE) ## Bash Reminder: Some commands take std-in while others directly take arguments. std-in basically are input-files.
ngpus=$((ngpus_per_node * num_nodes))

export MASTER_ADDR="localhost"
# export MASTER_ADDR=$(host $master_node | awk '{print $4}') ## Q. Only works on aurora for some reason? 
export MASTER_PORT="12345"
## TODO: Impelment DP and ZERO
# export SP=2 ## Ulysses World Size
# export ZERO=3 ## DP: ZERo=0

echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo ngpus_per_node: $ngpus_per_node
echo num_nodes: $num_nodes
echo ngpus: $ngpus
echo SP: $SP

mpiexec -ppn $ngpus_per_node -n $ngpus python $SCRIPT_DIR/holistic_ulysses.py |& tee $SCRIPT_DIR/test.log

## alternatively (torchrun automatically sets up of MASTER_ADDR and MASTER_PORT): 
## torchrun --nproc-per-node 12 /lus/flare/projects/Aurora_deployment/eugene/Microbenchmark/ml_communications/in_context_benchmarks/ulysses_benchmark.py


    # try:
    #     dist.all_to_all_single(out, input, group=group, input_split_sizes=input_splits, output_split_sizes=output_splits)
    #     print(f"good input.shape: {input.shape}\n"
    #             f"out.shape: {out.shape}\n"
    #             f"good is_first_all2all: {is_first_all2all}\n"
    #             f"good input.device: {input.device}\n"
    #             f"good RANK: {RANK}\n",
    #             f"good input.is_contiguous: {input.is_contiguous()}",
    #             f"good group: {group}",
    #             f"good input_splits: {input_splits}",
    #             f"good output_splits: {output_splits}",
    #             flush=True)
    # except:
    #     # zeros = torch.zeros_like(input)
    #     # zeros_out = torch.zeros_like(out)
    #     # dist.all_to_all_single(zeros_out, zeros, group=group, input_split_sizes=input_splits, output_split_sizes=output_splits)

    #     # assert input.is_cuda, "input.is_cuda"
    #     # assert not input.is_sparse, "input.is_sparse"

    #     print(f"input.shape: {input.shape}\n"
    #             f"out.shape: {out.shape}\n"
    #             f"is_first_all2all: {is_first_all2all}\n"
    #             f"input.device: {input.device}\n"
    #             f"RANK: {RANK}\n",
    #             f"input.is_contiguous: {input.is_contiguous()}",
    #             f"group: {group}",
    #             f"input_splits: {input_splits}",
    #             f"output_splits: {output_splits}",
    #             flush=True)
    #     # raise e
    #     raise KeyboardInterrupt()
