#!/bin/bash
## This wrapper should be used with unitrace to trace in the case of larger than
## 4 Nodes. The script is set up to trace rank 0 of first 4 Nodes in the case of
## profiling a job running on larger than 4 nodes.
FNAME_EXT=$(basename "$2")
FNAME="${FNAME_EXT%%.*}"
 
NNODES=`wc -l < $PBS_NODEFILE`

WORK_DIR=/home/hossainm/ml_communications/in_context_benchmarks/llm_benchmarks
UNITRACE_DIR=/opt/aurora/24.180.3/support/tools/pti-gpu/d3639de
UNITRACE_LIB=${UNITRACE_DIR}/lib64
UNITRACE_BIN=${UNITRACE_DIR}/bin
UNITRACE_EXE=${UNITRACE_BIN}/unitrace
DTAG=$(date +%F_%H%M%S)
UNITRACE_OUTDIR=${WORK_DIR}/run_scripts/outdir_aurora/logs/unitrace_profiles/tp_no_sp_pt_ddp_json_n${NNODES}_${DTAG}/${FNAME}_n${NNODES}_${DTAG}
mkdir -p ${UNITRACE_OUTDIR}
#UNITRACE_OPTS=" --chrome-sycl-logging  --chrome-mpi-logging  --chrome-ccl-logging  --chrome-dnn-logging --chrome-kernel-logging --chrome-device-logging --chrome-no-thread-on-device --chrome-no-engine-on-device --output-dir-path ${ITRACEOUTDIR}  --output ${ITRACEOUTDIR}/${FNAME}-ut "
UNITRACE_OPTS=" --ccl-summary-report --chrome-mpi-logging --chrome-sycl-logging \
--chrome-device-logging \
--chrome-ccl-logging --chrome-call-logging --chrome-dnn-logging --device-timing --host-timing \
--output-dir-path ${UNITRACE_OUTDIR} --output ${UNITRACE_OUTDIR}/UNITRACE_${FNAME}_n${NNODES}_${DTAG}.txt "

 
export LD_LIBRARY_PATH=${UNITRACE_LIB}:${UNITRACE_BIN}:$LD_LIBRARY_PATH
 
# Use $PMI_RANK for MPICH and $SLURM_PROCID with srun.
#PROFRANK=$(( ${NNODES}-1 ))
#PROFRANK=$(( $PALS_LOCAL_SIZE-1 ))
PROFRANK=0
RANKCUTOFF=48
#if [[ $PMIX_RANK -eq $PROFRANK ]]; then
#  echo "On rank $PMIX_RANK, collecting traces "
#  $UNITRACE_EXE $UNITRACE_OPTS "$@"
#else
#  "$@"
#fi
if [[ $PALS_LOCAL_RANKID -eq $PROFRANK ]] && [[ $PMIX_RANK -lt $RANKCUTOFF ]]; then
  echo "On rank $PMIX_RANK, collecting traces "
  $UNITRACE_EXE $UNITRACE_OPTS "$@"
else
  "$@"
fi
