#!/bin/bash -x
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir
#PBS -j oe
#PBS -N FP32_SP 

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
PRECISION="float32"
TRIAL=1

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module use /soft/modulefiles/
module load conda/2024-04-29
conda activate 

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
#export NCCL_NET="AWS Libfabric"
#export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="

RUN_ID=polaris_tensor_parallel_ENV_PHB_TP${TP_DEGREE}_SP_TIMING_LOOPS${TIMING_LOOPS}_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

echo "${RUN_ID}"


echo "$(timestamp): Before mpiexec."

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind=verbose,list:0:8:16:24 \
python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization.py \
-tp_degree=${TP_DEGREE} -sp_switch --warmup_iterations ${WARMUPS} --iterations ${TIMING_LOOPS} --precision ${PRECISION} \
--logging --log_directory=${WORK_DIR}/run_scripts/outdir/logs --log_file=${RUN_ID}.log

echo "$(timestamp): Finished the workload."

