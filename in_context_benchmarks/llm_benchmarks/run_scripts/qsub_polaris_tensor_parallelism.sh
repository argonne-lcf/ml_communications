#!/bin/bash -x
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir
#PBS -j oe
#PBS -N LLM_test_8x12 

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR=/home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 

TP_DEGREE=4
TIMING_LOOPS=4
WARMUPS=4
PRECISION="bfloat16"
TRIAL=1

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module use /soft/modulefiles/
module load conda/2024-04-29
conda activate 

# To prevent error message related to AWS Libfabric not found
unset NCCL_COLLNET_ENABLE NCCL_CROSS_NIC NCCL_NET NCCL_NET_GDR_LEVEL

echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="

RUN_ID=polaris_tensor_parallel_TP${TP_DEGREE}_SP_TIMING_LOOPS${TIMING_LOOPS}_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

echo "${RUN_ID}"


echo "$(timestamp): Before mpiexec."

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer \
python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization.py \
-tp_degree=${TP_DEGREE} -sp_switch --warmup_iterations ${WARMUPS} --iterations ${TIMING_LOOPS} --precision ${PRECISION} \
--logging --log_directory=${WORK_DIR}/run_scripts/outdir/logs --log_file=${RUN_ID}.log

echo "$(timestamp): Finished the workload."

