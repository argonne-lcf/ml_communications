#! /bin/bash

master_node=$(head -1 $PBS_NODEFILE)
export MASTER_ADDR=$(host $master_node | awk '{print $4}')
export MASTER_PORT="12345"

ngpus_per_node=12
## Reminder: Some commands take std-in while others directly take arguments. std-in basically are input-files.
num_nodes=$(wc -l < $PBS_NODEFILE) 
ngpus=$((ngpus_per_node * num_nodes))

echo ngpus_per_node: $ngpus_per_node
echo num_nodes: $num_nodes
echo ngpus: $ngpus

SCRIPT_PTH=/lus/flare/projects/Aurora_deployment/eugene/Microbenchmark/ml_communications/in_context_benchmarks/ulysses_benchmark.py ## has to be abs_path
SCRIPT_DIR=$(dirname $SCRIPT_PTH)
mpiexec -ppn $ngpus_per_node -n $ngpus python $SCRIPT_PTH |& tee $SCRIPT_DIR/test.log

## alternatively (torchrun doesn't require set up of MASTER_ADDR and MASTER_PORT): 
## torchrun --nproc-per-node 12 /lus/flare/projects/Aurora_deployment/eugene/Microbenchmark/ml_communications/in_context_benchmarks/ulysses_benchmark.py