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
#PBS -N ARDC_PT 

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

BENCH_DIR=/home/hossainm/ml_communications/hardware_benchmarks/frameworks/pytorch/ddp/oneCCL
LOG_WRAPPER=${BENCH_DIR}/log_wrapper.sh 

TRIAL=1
#MSG=1073741824 ## ~1.07 GB per Rank, if 4 ranks
#MSG=536870912 ## ~2.15 GB per rank
MSG=1073741824 ## ~2.15 GB per rank, BF16
#MSG=536870912 ## ~1.07 GB per rank, BF16

#ALGO=Ring

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

#module load frameworks/2024.2.1_u1

module unload oneapi/eng-compiler/2024.07.30.002
module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
module use /soft/preview/pe/24.347.0-RC2/modulefiles
module add oneapi/release
module use /lus/flare/projects/Aurora_deployment/datascience/software/aurora_fw_2025.0.1_u1_test_lus-umd1077p18/modulefiles/

export FI_MR_CACHE_MONITOR=disabled

## To check if things are reasonable
echo "loaded modules"
module -t list
module load frameworks-dev
echo "loaded modules"
module -t list

#export CCL_CONFIGURATION_PATH=""
#export CCL_CONFIGURATION=cpu_gpu_dpcpp
#export CCL_ROOT="/lus/flare/projects/Aurora_deployment/datascience/software/ccl_2021.15/oneCCL/build_2021p15/"
#export LD_LIBRARY_PATH=/lus/flare/projects/Aurora_deployment/datascience/software/ccl_2021.15/oneCCL/build_2021p15/lib:$LD_LIBRARY_PATH
#export CPATH=/lus/flare/projects/Aurora_deployment/datascience/software/ccl_2021.15/oneCCL/build_2021p15/include:$CPATH
#export LIBRARY_PATH=/lus/flare/projects/Aurora_deployment/datascience/software/ccl_2021.15/oneCCL/build_2021p15/lib:$LIBRARY_PATH
source /lus/flare/projects/Aurora_deployment/datascience/software/ccl_2021.15/oneCCL/build_2021p15/env/vars.sh --ccl-bundled-mpi=no

#export CCL_LOG_LEVEL=debug

## For TP=2, PPN=1
## Special case for testing
#export CPU_AFFINITY="list:0-2,4-7,104-111"
#export HOROVOD_THREAD_AFFINITY="4"
#export CCL_WORKER_AFFINITY="3"
#export MEM_BIND="list:2"
#export ZE_AFFINITY_MASK="0"

## For TP=2, PPN=2
# Within a GPU, tile-to-tile MDFI
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119"
#export HOROVOD_THREAD_AFFINITY="4,12"
#export CCL_WORKER_AFFINITY="3,11"
#export MEM_BIND="list:2:3"
#export ZE_AFFINITY_MASK="0,1"
#

## For TP=4, PPN=4
#export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135"
#export HOROVOD_THREAD_AFFINITY="4,12,20,28"
#export CCL_WORKER_AFFINITY="3,11,19,27"
#export MEM_BIND="list:2:2:3:3"
#export ZE_AFFINITY_MASK="0,1,2,3"

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


#RUN_ID=aurora_ALLREDUCE_CB08_ZE01_1GB_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")
RUN_ID=aurora_ALLREDUCE_2GB_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

#LOG_DIR=${WORK_DIR}/run_scripts/outdir/logs 

echo "${RUN_ID}"

echo "$(timestamp): Before mpiexec."

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} --mem-bind ${MEM_BIND} \
    python ${BENCH_DIR}/gpu_allreduce.py --tensor_dimension_1d=${MSG}

echo "$(timestamp): Finished the workload."

## clean up phase
conda deactivate
module unload frameworks-dev
module -t list
module restore
module -t list

