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
# module load frameworks
# . /c/Users/eugen/Codes/ml_communications/venv/Scripts/activate

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}
echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR="/lus/flare/projects/Aurora_deployment/eku/ml_communications/in_context_benchmarks/2D_parallelism"
LOG_WRAPPER="$WORK_DIR/log_wrapper.sh"
TP_DEGREE=12
ULYSSES_DEGREE=1
WARMUP_ITERS=1
NUM_ITERS=5  # Switch back to N_TIMING_LOOP?
PRECISION="float32"
N_LAYERS=80  # FIXME: 80
DEVICE="xpu"  # {cpu, cuda, xpu}
# DEVICE="cpu"  # {cpu, cuda, xpu}
NNODES=$(wc -l < $PBS_NODEFILE)
# NNODES=1
NRANKS_PER_NODE=12
NRANKS=$((NNODES * NRANKS_PER_NODE))

RUN_ID=unified_$(date +"%Y-%m-%d_%H-%M-%S")
if [[ $DEVICE == "cpu" ]]; then
  let HIDDEN_DIM=NRANKS*2
  PYTHON_ARGS="\
    -s $NRANKS \
    -d $HIDDEN_DIM \
    --device ${DEVICE} \
    --ulysses ${ULYSSES_DEGREE} \
    -TP ${TP_DEGREE} \
    --sequence-parallel-switch \
    --warmup-iter ${WARMUP_ITERS} \
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
  # let HIDDEN_DIM=NRANKS*2
  # PYTHON_ARGS="\
  #   -s $NRANKS \
  #   -d $HIDDEN_DIM \
  #   --device ${DEVICE} \
  #   --ulysses ${TP_DEGREE} \
  #   -TP 1 \
  #   --sequence-parallel-switch \
  #   --warmup-iter ${WARMUP_ITERS} \
  #   --iterations=${NUM_ITERS} \
  #   --precision ${PRECISION} \
  #   -n_layers 2 \
  #   -bucket 50
  #   --logging \
  #   --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs \
  #   --log_file=${RUN_ID}.log \
  #   --head-count 12 \
  #   --include-flash-attention
  # "
    # --use-zero3
    # -TP ${TP_DEGREE} \
    # --ulysses ${ULYSSES_DEGREE} \
  PYTHON_ARGS="\
    --device ${DEVICE} \
    --ulysses 1 \
    -TP ${TP_DEGREE} \
    --sequence-parallel-switch \
    -n_layers $N_LAYERS \
    --warmup-iter ${WARMUP_ITERS} \
    --iterations=${NUM_ITERS} \
    --precision ${PRECISION} \
    --logging \
    --log_directory=${WORK_DIR}/run_scripts/outdir_aurora/logs \
    --log_file=${RUN_ID}.log \
    --head-count 12 \
    --include-flash-attention
  "
    # --use-zero3
fi

export USE_TORCHRUN=0

if [[ $USE_TORCHRUN -eq 1 ]]; then
  echo "using torchrun"
  torchrun --nproc-per-node=$NRANKS_PER_NODE \
    "$WORK_DIR/tp_ulysses_zero3.py" \
    $PYTHON_ARGS
  echo PYTHON_ARGS: $PYTHON_ARGS
    # torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR/send_recv_bench.py"
    # torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\train.py"
else
  export MASTER_ADDR='localhost'
  export MASTER_PORT=24500
  echo "using MPIEXEC"
  mpiexec --envall --verbose -n $NRANKS -ppn $NRANKS_PER_NODE \
    python "$WORK_DIR/tp_ulysses_zero3.py" $PYTHON_ARGS
fi

echo "$(timestamp): Finished the workload."
echo "logged at ${WORK_DIR}/run_scripts/outdir_aurora/logs/${RUN_ID}.log"

## Key changes made from the original program
# 1. timed kernels using a wrapper-like function 'timed'
# 2. Changed warm up to be iter based.
# 3. change how groups are initialzied
# 4. Change to timing without blocking for async comm. 
#    synchronization calls will corrupt time
# 5. Implement Ulysses, Zero3
# 6. made timing async
# 7. etc...