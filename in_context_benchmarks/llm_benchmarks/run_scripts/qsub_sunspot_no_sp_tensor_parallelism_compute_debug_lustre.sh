#!/bin/bash -x
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:10:00
#PBS -q validation
#PBS -A Aurora_deployment
#PBS -l filesystems=home:gila
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir_sunspot
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir_sunspot
#PBS -j oe
#PBS -N lustre_tp 

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR=/home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 
CACHE_SOFT=${WORK_DIR}/cache_soft.py

TP_DEGREE=12
#WARMUPS=1
TIMING_LOOPS=4
PRECISION="float32"
N_LAYERS=1
TRIAL=4

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

#echo "$(timestamp): Start the moving to tmp process on ${NNODES} Nodes"
#mkdir -p /tmp/${USER}/fw_tar
#cp /lus/flare/projects/Aurora_deployment/datascience/software/aurora_fw_2025.0.1_u1_test_lustre-feb8.tar.gz /tmp/${USER}/fw_tar

#module load frameworks

#mpiexec --transfer --pmi=pmix -np $NNODES --ppn 1 ${CACHE_SOFT} \
#      --src /tmp/${USER}/fw_tar/aurora_fw_2025.0.1_u1_test_lustre-feb8.tar.gz \
#      --dst /tmp/aurora_fw_2025.0.1_u1_test_lustre-feb8.tar.gz --d
#
#module unload frameworks

#module restore
#echo "$(timestamp): End the moving to tmp process on ${NNODES} Nodes"


# Trying the frameworks-dev module
#module load frameworks/2024.2.1_u1
#
module unload oneapi/eng-compiler/2024.07.30.002
module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
module use /soft/preview/pe/24.347.0-RC2/modulefiles
module add oneapi/release
module use /home/hossainm/frameworks_factory/aurora_fw_2025.0.1_u1_test_lus-umd1077p18/modulefiles
echo "loaded modules"
module -t list
module load frameworks-dev
echo "loaded modules"
module -t list

export FI_MR_CACHE_MONITOR=disabled

#export CPU_AFFINITY="verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"
export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:32-34,36-39,136-143:40-42,44-47,144-151:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187:84-86,88-91,188-195:92-94,96-99,196-203"
#export CPU_AFFINITY="verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
export HOROVOD_THREAD_AFFINITY="4,12,20,28,36,44,56,64,72,80,88,96"
#export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"
export CCL_WORKER_AFFINITY="3,11,19,27,35,43,55,63,71,79,87,95"
export MEM_BIND="list:2:2:2:2:2:2:3:3:3:3:3:3"

export CCL_ATL_TRANSPORT=mpi
export CCL_LOG_LEVEL=debug

#echo "========= ENVIRONMENT VARIABLES ======="
#env
#echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="

RUN_ID=sunspot_lustre_tensor_parallel_TP${TP_DEGREE}_NO_SP_LAYERS${N_LAYERS}_TIMING_LOOPS${TIMING_LOOPS}_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

echo "${RUN_ID}"


echo "$(timestamp): Before mpiexec."

mpiexec --pmi=pmix -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} --mem-bind ${MEM_BIND} \
${LOG_WRAPPER} python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization_debug.py -dvc "xpu" \
-tp_degree=${TP_DEGREE}  --barrier --iterations=${TIMING_LOOPS} --precision ${PRECISION} -n_layers ${N_LAYERS} \
--logging --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs --log_file=${RUN_ID}.log

echo "$(timestamp): Finished the workload."

## Clean up

# Unload will only deactivate the CONDA and remove paths related to CONDA
module unload frameworks-dev
module -t list
# restore to get back to the default state with other ad-hoc paths removed
module restore
module -t list ## Check the loaded modules

