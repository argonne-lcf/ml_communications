#!/bin/bash -x
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:eagle
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir
#PBS -j oe
#PBS -N FP32_NOSP 

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
#WARMUPS=4
PRECISION="float32"
N_LAYERS=1
TRIAL=1
SOCKET=hsn

ALGO=Ring

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module use /soft/modulefiles/
module load conda/2024-04-29
conda activate 

## The following is the old set, used for all the runs before Jan 17, 2025
## with NCCL_NET, and LD_LIBRARY_PATHS commented out (i.e. not being loaded)
#export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_CROSS_NIC=1
#export NCCL_COLLNET_ENABLE=1
#export NCCL_NET="AWS Libfabric"
#export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
#export FI_CXI_DISABLE_HOST_REGISTER=1
#export FI_MR_CACHE_MONITOR=userfaultfd
#export FI_CXI_DEFAULT_CQ_SIZE=131072

## The following are the new set. Testing if these resolve the AWS plugin hang
## The recommendation is using AWS-V1.6.0
## I am trying AWS-V1.9.1
#export AWS_DIR=/soft/libraries/aws-ofi-nccl/v1.6.0/
export AWS_DIR=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=$AWS_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH

export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RX_MATCH_MODE=software
export FI_CXI_REQ_BUF_SIZE=16MB

export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
export FI_CXI_RDZV_THRESHOLD=2000

export NCCL_ALGO=${ALGO}

export NCCL_SOCKET_IFNAME=${SOCKET}

#export NCCL_DEBUG=INFO

#unset NCCL_COLLNET_ENABLE NCCL_CROSS_NIC NCCL_NET NCCL_NET_GDR_LEVEL

echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="

#CPU_BIND=verbose,list:0:8:16:24 ## for PPN 4
#CPU_BIND=verbose,list:0:16 ## for PPN2
CPU_BIND=verbose,list:0:24 ## for PPN2
#CPU_BIND=verbose,list:0 ## for PPN1



RUN_ID=polaris_tensor_parallel_All_Ranks_CB024_Barrier_Sync_Socket_${SOCKET}_AWS1p9p1_ENV_PHB_TP${TP_DEGREE}_NO_SP_NCCL_ALGO${ALGO}_NOWARMUPS_LAYERS${N_LAYERS}_TIMING_LOOPS${TIMING_LOOPS}_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR=${WORK_DIR}/run_scripts/outdir/logs 

echo "${RUN_ID}"

echo "$(timestamp): Before mpiexec."

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_BIND} \
python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization_debug.py -n_layers ${N_LAYERS} \
-tp_degree=${TP_DEGREE} --barrier --iterations=${TIMING_LOOPS} --precision ${PRECISION} \
--logging --log_directory=${LOG_DIR} --log_file=${RUN_ID} 

echo "$(timestamp): Finished the workload."

