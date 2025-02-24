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
# export TZ='/usr/share/zoneinfo/US/Central'
# timestamp() {
#   date +"%Y-%m-%d %H:%M:%S" # current time
# }
# echo "$(timestamp): Start of the Run, after exporting TZ Central"

SCRIPT_PTH="/eagle/datascience/eku/ml_communications/in_context_benchmarks/2D_parallelism/qsub_tp_zero3_polaris.sh"
WORK_DIR=$(dirname $SCRIPT_PTH | xargs realpath)
# LOG_WRAPPER="$WORK_DIR/log_wrapper.sh"
NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=4
NRANKS=$((NNODES * NRANKS_PER_NODE))
log_fpth="$WORK_DIR/logs/unified_$(date +"%Y-%m-%d_%H-%M-%S").log"

MODEL_PARALLELISM_DEGREE=$NRANKS_PER_NODE
WARMUP_ITERS=1
NUM_ITERS=5  # Switch back to N_TIMING_LOOP?
PRECISION="float32"
N_LAYERS=80  # FIXME: 80
DEVICE="cuda"  # {cpu, cuda, xpu}
# DEVICE="cpu"  # {cpu, cuda, xpu}

# export AWS_DIR=/soft/libraries/aws-ofi-nccl/v1.6.0/
# export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_CROSS_NIC=1
# export NCCL_COLLNET_ENABLE=1
# export NCCL_SOCKET_IFNAME=hsn
# export NCCL_NET="AWS Libfabric"
# export LD_LIBRARY_PATH=$AWS_DIR/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH

# export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_MR_CACHE_MONITOR=userfaultfd
# export FI_CXI_DEFAULT_CQ_SIZE=131072
# export FI_CXI_DEFAULT_TX_SIZE=131072
# export FI_CXI_RDZV_PROTO=alt_read
# export FI_CXI_RX_MATCH_MODE=software
# export FI_CXI_REQ_BUF_SIZE=16MB

# export FI_CXI_RDZV_GET_MIN=0
# export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
# export FI_CXI_RDZV_THRESHOLD=2000

# if [[ $DEVICE == "cpu" ]]; then
#   exit 1
#   let HIDDEN_DIM=NRANKS*2
#   PYTHON_ARGS="\
#     -s $NRANKS \
#     -d $HIDDEN_DIM \
#     --device ${DEVICE} \
#     --ulysses ${ULYSSES_DEGREE} \
#     -TP ${TP_DEGREE} \
#     --sequence-parallel-switch \
#     --warmup-iter ${WARMUP_ITERS} \
#     --iterations=${NUM_ITERS} \
#     --precision ${PRECISION} \
#     -n_layers 2 \
#     -bucket 50 \
#     --logging \
#     --log_fpth $log_fpth \
#     --head-count 12 \
#     --include-flash-attention \
#     --use-zero3
#   "
# else
  # -TP $MODEL_PARALLELISM_DEGREE \
PYTHON_ARGS="\
  --device ${DEVICE} \
  --ulysses $MODEL_PARALLELISM_DEGREE \
  --sequence-parallel-switch \
  -n_layers $N_LAYERS \
  --warmup-iter ${WARMUP_ITERS} \
  --iterations=${NUM_ITERS} \
  --precision ${PRECISION} \
  --logging \
  --log_fpth $log_fpth \
  --head-count 12 \
  --include-flash-attention \
  --trace
"
    # -TP ${TP_DEGREE} \
    # --ulysses ${ULYSSES_DEGREE} \
    # --use-zero3
# fi


# export USE_TORCHRUN=0
# if [[ $USE_TORCHRUN -eq 1 ]]; then
#   echo "using torchrun"
#   exit 1
#   torchrun --nproc-per-node=$NRANKS_PER_NODE \
#     "$WORK_DIR/tp_ulysses_zero3.py" \
#     $PYTHON_ARGS
#   echo PYTHON_ARGS: $PYTHON_ARGS
#     # torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR/send_recv_bench.py"
#     # torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\train.py"
# else
echo "using MPIEXEC"
export MASTER_ADDR='localhost'
export MASTER_PORT="42344"
mpiexec --envall --verbose -n $NRANKS -ppn $NRANKS_PER_NODE \
  python "$WORK_DIR/tp_ulysses_zero3.py" $PYTHON_ARGS
# fi

# # echo "$(timestamp): 
# echo "Finished the workload."
# echo "logged at $log_fpth"

# ## Key changes made from the original program
# # 1. timed kernels using a wrapper-like function 'timed'
# # 2. Changed warm up to be iter based.
# # 3. change how groups are initialzied
# # 4. Change to timing without blocking for async comm. 
# #    synchronization calls will corrupt time
# # 5. Implement Ulysses, Zero3
# # 6. made timing async
# # 7. etc...