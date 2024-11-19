#!/bin/bash +x

SCRIPT_PTH=/eagle/datascience/eku/ml_communications/in_context_benchmarks/vit_benchmarks/qsub_ulysses.sh ## use abs_path
SCRIPT_DIR=$(dirname $SCRIPT_PTH)

echo "how to activate base env in a script?"

master_node=$(head -1 $PBS_NODEFILE)
## TODO: make below agnostic to aurora, sunspot, polaris
ngpus_per_node=4
# ngpus_per_node=12
num_nodes=$(wc -l < $PBS_NODEFILE) ## Bash Reminder: Some commands take std-in while others directly take arguments. std-in basically are input-files.
ngpus=$((ngpus_per_node * num_nodes))

export MASTER_ADDR="localhost"
# export MASTER_ADDR=$(host $master_node | awk '{print $4}') ## Q. Only works on aurora for some reason? 
export MASTER_PORT="12345"
export SP=2 ## Ulysses World Size
export ZERO=3 ## DP: ZERo=0

echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
echo ngpus_per_node: $ngpus_per_node
echo num_nodes: $num_nodes
echo ngpus: $ngpus
echo SP: $SP

## Pure Communication Ulysses
mpiexec -ppn $ngpus_per_node -n $ngpus python $SCRIPT_DIR/pure_comm_ulysses.py --SP $SP --zero $ZERO |& tee $SCRIPT_DIR/test.log

## alternatively (torchrun automatically sets up of MASTER_ADDR and MASTER_PORT): 
## torchrun --nproc-per-node 12 /lus/flare/projects/Aurora_deployment/eugene/Microbenchmark/ml_communications/in_context_benchmarks/ulysses_benchmark.py