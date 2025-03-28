#!/bin/bash -x
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir_aurora
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir_aurora
#PBS -j oe
#PBS -N ARDC_PT 

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR=/home/hossainm/ml_communications/hardware_benchmarks/frameworks/pytorch/mpi4py
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 

TRIAL=1
#MSG=1073741824 ## ~1.07 GB per Rank, if 4 ranks
#MSG=536870912 ## ~2.15 GB per rank
#MSG=1073741824 ## ~2.15 GB per rank, BF16 -- doesn't fit in int, MPI error
MSG=536870912 ## ~1.07 GB per rank, BF16

#ALGO=Ring

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=2

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module load frameworks/2024.2.1_u1

## For TP=2, PPN=1
## Special case for testing
#export CPU_AFFINITY="list:0-2,4-7,104-111"
#export HOROVOD_THREAD_AFFINITY="4"
#export CCL_WORKER_AFFINITY="3"
#export MEM_BIND="list:2"
#export ZE_AFFINITY_MASK="0"

## For TP=2, PPN=2
# Within a GPU, tile-to-tile MDFI
export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119"
export HOROVOD_THREAD_AFFINITY="4,12"
export CCL_WORKER_AFFINITY="3,11"
export MEM_BIND="list:2:3"
export ZE_AFFINITY_MASK="0,1"

## For TP=12, PPN=12
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:32-34,36-39,136-143:40-42,44-47,144-151:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187:84-86,88-91,188-195:92-94,96-99,196-203"
#export HOROVOD_THREAD_AFFINITY="4,12,20,28,36,44,56,64,72,80,88,96"
#export CCL_WORKER_AFFINITY="3,11,19,27,35,43,55,63,71,79,87,95"
#export MEM_BIND="list:2:2:2:2:2:2:3:3:3:3:3:3"
#export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"


echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="


#RUN_ID=aurora_mpi4py_ALLREDUCE_CB0_ZE0_1GB_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")
RUN_ID=aurora_mpi4py_ALLREDUCE_1GB_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")
#LOG_DIR=${WORK_DIR}/run_scripts/outdir/logs 

echo "${RUN_ID}"

echo "$(timestamp): Before mpiexec."

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} --mem-bind ${MEM_BIND} \
    python ${WORK_DIR}/gpu_allreduce.py --tensor_dimension_1d=${MSG}

echo "$(timestamp): Finished the workload."

