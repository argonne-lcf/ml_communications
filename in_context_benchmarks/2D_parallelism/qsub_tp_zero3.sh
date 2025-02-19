#!/bin/bash -x
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q lustre_scaling
#PBS -A Aurora_deployment
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -e /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/errordir_aurora
#PBS -o /home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks/run_scripts/outdir_aurora
#PBS -j oe
#PBS -N LLM_test_8x12 

## FIXME: local cpu environment
. /c/Users/eugen/Codes/ml_communications/venv/Scripts/activate

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}
echo "$(timestamp): Start of the Run, after exporting TZ Central"

# module load frameworks/2024.2.1_u1
WORK_DIR='C:\Users\eugen\Codes\ml_communications\in_context_benchmarks\2D_parallelism'
# WORK_DIR="/Users/eugen/Areas/Tutorials/ml_communications/in_context_benchmarks/llm_benchmarks"
# WORK_DIR="/lus/flare/projects/Aurora_deployment/eku/ml_communications/in_context_benchmarks/llm_benchmarks"
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 
TP_DEGREE=12
WARMUP_ITERS=1
NUM_ITERS=5  # Switch back to N_TIMING_LOOP?
PRECISION="float32"
N_LAYERS=80
DEVICE="cpu"  # {cpu, cuda, xpu}
# DEVICE="xpu"  # {cpu, cuda, xpu}
# TRIAL=2
# MPI and OpenMP settings
# NNODES=`wc -l < $PBS_NODEFILE`
NNODES=1
NRANKS_PER_NODE=12
let NRANKS=NNODES*NRANKS_PER_NODE
export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:32-34,36-39,136-143:40-42,44-47,144-151:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187:84-86,88-91,188-195:92-94,96-99,196-203"
export HOROVOD_THREAD_AFFINITY="4,12,20,28,36,44,56,64,72,80,88,96"
export CCL_WORKER_AFFINITY="3,11,19,27,35,43,55,63,71,79,87,95"
export MEM_BIND="list:2:2:2:2:2:2:3:3:3:3:3:3"

echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES =======\n"
echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES ==============\n"
RUN_ID=aurora_tensor_parallel_TP${TP_DEGREE}_NO_SP_${N_LAYERS}LAYERS_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_$(date +"%Y-%m-%d_%H-%M-%S")
echo "${RUN_ID}"
echo "$(timestamp): Before mpiexec."

  # -n $NNODES \
  # -ppn 1 \
MPI_ARGS="\
  --pmi=pmix \
  -n ${NRANKS} \
  -ppn ${NRANKS_PER_NODE} \
  -l \
  --line-buffer \
  --cpu-bind ${CPU_AFFINITY} \
  --mem-bind ${MEM_BIND} \
  ${LOG_WRAPPER}
"
if [[ $DEVICE == "cpu" ]]; then
  let HIDDEN_DIM=NRANKS*2
  ## try sequence paralle
  ## try zero3
  PYTHON_ARGS="\
    -s $NRANKS \
    -d $HIDDEN_DIM \
    --device ${DEVICE} \
    -TP ${TP_DEGREE} \
    --sequence-parallel-switch
    --warmup_iterations ${WARMUP_ITERS} \
    --iterations=${NUM_ITERS} \
    --precision ${PRECISION} \
    -n_layers 2 \
    -bucket 50
    --logging \
    --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs \
    --log_file=${RUN_ID}.log \
    --head-count 12 \
    --include-flash-attention
    --use-zero3
  "
else
  PYTHON_ARGS="\
    --device ${DEVICE} \
    -TP=${TP_DEGREE} \
    --warmup_iterations ${WARMUP_ITERS} \
    --iterations=${NUM_ITERS} \
    --precision ${PRECISION} \
    -n_layers ${N_LAYERS} \
    --logging \
    --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs \
    --log_file=${RUN_ID}.log
  "
  echo "Only train on cpu until the clusters are up"
  exit 1;
fi
# mpiexec $MPI_ARGS \
#   python ${WORK_DIR}/tensor_parallel_with_gradient_synchronization.py \
#   $PYTHON_ARGS
# mpiexec $MPI_ARGS \
#   python ${WORK_DIR}/unified_parallelism.py \
#   $PYTHON_ARGS

# RDZV_HOST=$(hostname)
# RDZV_PORT=$RANDOM
# ip link show
# export CCL_SOCKET_IFNAME="enp0s31f6"
# --nnodes=1
# mpiexec $

# torchrun --standalone --rdzv-backend=c10d --rdzv-endpoint="$RDZV_HOST:$RDZV_PORT" \
# mpiexec $MPI_ARGS python3 -m torch.distributed.run \
# mpiexec -n $NNODES -ppn 1 python3 -m torch.distributed.run \
#   --rdzv-backend=c10d --rdzv-endpoint="$RDZV_HOST:$RDZV_PORT" \
#   --nnodes=$NNODES --nproc-per-node=$NRANKS_PER_NODE \
#   "${WORK_DIR}/unified_parallelism.py" \
#   $PYTHON_ARGS

torchrun --nproc-per-node=$NRANKS \
  "${WORK_DIR}\tp_zero3.py" \
  $PYTHON_ARGS

echo "$(timestamp): Finished the workload."

## Key changes made from the original program
# sync+time -> sync_and_time() - merged func
# timed kernels using a wrapper-like function 'timed'
# 2. Changed warm up to be iter based.
# 3. change how groups are initialzied

## 
# mpiexec unified_parallelism.py
# Questions for myself:
# How is group initialized
# Where are the compute and comm?
# how is the timing done?

## TODO: Assert Ulysses + TP (don't want to think about it)