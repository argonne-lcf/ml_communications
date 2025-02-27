#!/bin/bash -x
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir_aurora
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir_aurora
#PBS -j oe
#PBS -N TP96_R12

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR=/home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 

TP_DEGREE=96
#WARMUPS=1
TIMING_LOOPS=4
PRECISION="float32"
N_LAYERS=80
TRIAL=1

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module load frameworks/2024.2.1_u1

## For TP=2, PPN=2
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119"
#export HOROVOD_THREAD_AFFINITY="4,12"
#export CCL_WORKER_AFFINITY="3,11"
#export MEM_BIND="list:2:3"
#export ZE_AFFINITY_MASK="0,2"

## For TP=4, PPN=4
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135"
#export HOROVOD_THREAD_AFFINITY="4,12,20,28"
#export CCL_WORKER_AFFINITY="3,11,19,27"
#export MEM_BIND="list:2:2:3:3"
#export ZE_AFFINITY_MASK="0,1,2,3"

## For TP=6, PPN=6
# CPU cores from 2 sockets, Physical core 0-51:socket 1, Physical Core 52-103: socket 2 
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179"
#export HOROVOD_THREAD_AFFINITY="4,12,20,56,64,72"
#export CCL_WORKER_AFFINITY="3,11,19,55,63,71"
#export MEM_BIND="list:2:2:2:3:3:3"
## Symmetric division, tile 0s of GPUs used only
#export ZE_AFFINITY_MASK="0,2,4,6,8,10"

## For TP=8,16,24 PPN=8
# CPU cores from 2 sockets, Physical core 0-51:socket 1, Physical Core 52-103: socket 2 
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187"
#export HOROVOD_THREAD_AFFINITY="4,12,20,28,56,64,72,80"
#export CCL_WORKER_AFFINITY="3,11,19,27,55,63,71,79"
#export MEM_BIND="list:2:2:2:2:3:3:3:3"
## Both tiles of each GPUs used to make 8 tiles
#export ZE_AFFINITY_MASK="0,1,2,3,6,7,8,9"

## For TP=12, PPN=12
export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:32-34,36-39,136-143:40-42,44-47,144-151:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187:84-86,88-91,188-195:92-94,96-99,196-203"
export HOROVOD_THREAD_AFFINITY="4,12,20,28,36,44,56,64,72,80,88,96"
export CCL_WORKER_AFFINITY="3,11,19,27,35,43,55,63,71,79,87,95"
export MEM_BIND="list:2:2:2:2:2:2:3:3:3:3:3:3"
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"


echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="

## RUN ID for R8 setup, NIC balanced case
#RUN_ID=aurora_tensor_parallel_CB08162452606876_ZED01236789_TP${TP_DEGREE}_NO_SP_LAYERS${N_LAYERS}_TIMING_LOOPS${TIMING_LOOPS}_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")
## RUN ID for R12 setup, Regular, NIC imbalanced case
RUN_ID=aurora_tensor_parallel_TP${TP_DEGREE}_NO_SP_LAYERS${N_LAYERS}_TIMING_LOOPS${TIMING_LOOPS}_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")


echo "${RUN_ID}"


echo "$(timestamp): Before mpiexec."

mpiexec --pmi=pmix -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} --mem-bind ${MEM_BIND} \
${LOG_WRAPPER} python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization_debug.py -dvc "xpu" \
-tp_degree=${TP_DEGREE}  --barrier --iterations=${TIMING_LOOPS} --precision ${PRECISION} -n_layers ${N_LAYERS} \
--logging --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs/tp_sweep_for_sc25 --log_file=${RUN_ID}.log

echo "$(timestamp): Finished the workload."
