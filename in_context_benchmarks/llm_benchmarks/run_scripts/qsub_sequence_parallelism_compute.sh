#!/bin/bash -x
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -q lustre_scaling
#PBS -A Aurora_deployment
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -e /home/hossainm/alcf_4_ai_benchmarks/in_context_benchmarks/run_scripts/errordir
#PBS -o /home/hossainm/alcf_4_ai_benchmarks/in_context_benchmarks/run_scripts/outdir
#PBS -j oe
#PBS -N LLM_test_1x12 

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR=/home/hossainm/alcf_4_ai_benchmarks/in_context_benchmarks

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module load frameworks/2024.2.1_u1

#export CPU_AFFINITY="verbose,list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203"
export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:32-34,36-39,136-143:40-42,44-47,144-151:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187:84-86,88-91,188-195:92-94,96-99,196-203"
#export CPU_AFFINITY="verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
export HOROVOD_THREAD_AFFINITY="4,12,20,28,36,44,56,64,72,80,88,96"
#export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"
export CCL_WORKER_AFFINITY="3,11,19,27,35,43,55,63,71,79,87,95"
export MEM_BIND="list:2:2:2:2:2:2:3:3:3:3:3:3"

echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="


echo "$(timestamp): Before mpiexec."

mpiexec --no-vni --pmi=pmix -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} --mem-bind ${MEM_BIND} \
python ${WORK_DIR}/sequence_parallelism_compute.py 
