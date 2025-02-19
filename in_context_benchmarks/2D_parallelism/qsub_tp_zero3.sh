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

WORK_DIR='C:\Users\eugen\Codes\ml_communications\in_context_benchmarks\2D_parallelism'
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 
TP_DEGREE=1
ULYSSES_DEGREE=1
WARMUP_ITERS=1
NUM_ITERS=5  # Switch back to N_TIMING_LOOP?
PRECISION="float32"
N_LAYERS=80
DEVICE="cpu"  # {cpu, cuda, xpu}
# DEVICE="xpu"  # {cpu, cuda, xpu}
NNODES=1
NRANKS_PER_NODE=12
let NRANKS=NNODES*NRANKS_PER_NODE

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
  echo "Only train on cpu until the clusters are up"
  exit 1;
fi


torchrun --nproc-per-node=$NRANKS \
  "${WORK_DIR}\tp_zero3.py" \
  $PYTHON_ARGS
echo "$(timestamp): Finished the workload."

## Key changes made from the original program
# 1. timed kernels using a wrapper-like function 'timed'
# 2. Changed warm up to be iter based.
# 3. change how groups are initialzied
# 4. Change to timing without blocking for async comm which 
#    synchronization calls will corrupt the time
# 5. Implement Ulysses, Zero3
# 6. made timing async
# 7. etc...