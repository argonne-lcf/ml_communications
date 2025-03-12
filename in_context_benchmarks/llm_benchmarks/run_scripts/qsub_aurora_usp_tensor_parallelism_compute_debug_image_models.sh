#!/bin/bash -x
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir_aurora
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir_aurora
#PBS -j oe
#PBS -N USP12_R12

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR=/home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 

## SEQ=4608, HID=16384, N_LAYERS=126, Llama 405B
## SEQ=16896, HID=25872, N_LAYERS=128, Llama 1T+ w/ large sequence length
#
TP_DEGREE=12
SEQ=4608
HID=9216
#WARMUPS=1
TIMING_LOOPS=4
PRECISION="bfloat16"
N_LAYERS=1
#N_LAYERS=1
## Special flag for deterministic input of torch ones. The defaults is 
# torch.normal with std_dev 0.01 around mean 0.00
IN_TYPE="random"
#IN_TYPE="torch_ones"
BUCKET=1e9
TRIAL=10

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

## For TP=6, PPN=6, Setup 2
# CPU cores from 2 sockets, Physical core 0-51:socket 1, Physical Core 52-103: socket 2 
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:52-54,56-59,156-163:60-62,64-67,164-171"
#export HOROVOD_THREAD_AFFINITY="4,12,20,56,64,72"
#export CCL_WORKER_AFFINITY="3,11,19,55,63,71"
#export MEM_BIND="list:2:2:2:3:3:3"
## Asymmetric division, full GPUs used only - GPU 0, 1 and GPU 3
#export ZE_AFFINITY_MASK="0,1,2,3,6,7"

## For TP=8,16,24 PPN=8
# CPU cores from 2 sockets, Physical core 0-51:socket 1, Physical Core 52-103: socket 2 
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187"
#export HOROVOD_THREAD_AFFINITY="4,12,20,28,56,64,72,80"
#export CCL_WORKER_AFFINITY="3,11,19,27,55,63,71,79"
#export MEM_BIND="list:2:2:2:2:3:3:3:3"
## Both tiles of each GPUs used to make 8 tiles, GPU 3 and 5 remains unused
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

## RUN ID for PPN=2
#RUN_ID=aurora_tensor_parallel_CB08_ZE02_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_NO_SP_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

## RUN ID for PPN=4
#RUN_ID=aurora_tensor_parallel_CB081624_ZE0123_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_NO_SP_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

## RUN ID for PPN=6, Asymmetric GPU placement
#RUN_ID=aurora_tensor_parallel_CB0816245260_ZE012367_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_USP_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

## RUN ID for PPN=8, NIC balanced case
#RUN_ID=aurora_tensor_parallel_CB08162452606876_ZE01236789_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_NO_SP_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

## RUN ID for ULSS R12 setup, Regular, NIC imbalanced case
RUN_ID=aurora_tensor_parallel_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_USP_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

## RUN ID for JUST TP R12 setup, Regular, NIC imbalanced case
#RUN_ID=aurora_tensor_parallel_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_NO_SP_NO_USP_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

## RUN ID for JUST SP R12 setup, Regular, NIC imbalanced case
#RUN_ID=aurora_tensor_parallel_SEQ${SEQ}_HID${HID}_TP${TP_DEGREE}_SP_NO_USP_L${N_LAYERS}_TL${TIMING_LOOPS}_${PRECISION}_${IN_TYPE}_ELEM_${BUCKET}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

echo "${RUN_ID}"


echo "$(timestamp): Before mpiexec."

mpiexec --pmi=pmix -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} --mem-bind ${MEM_BIND} \
${LOG_WRAPPER} python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization_a2a_debug.py -dvc "xpu" \
-tp_degree=${TP_DEGREE}  --sequence_length=${SEQ} --hidden_dimension=${HID} --barrier --iterations=${TIMING_LOOPS} --precision ${PRECISION} -n_layers ${N_LAYERS} \
-bucket ${BUCKET} -ulysses_enable \
--logging --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs/usp_scratch --log_file=${RUN_ID}.log

echo "$(timestamp): Finished the workload."
