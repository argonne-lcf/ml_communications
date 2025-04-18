#!/bin/bash
num_gpus=12
#gpu_id=$((PALS_LOCAL_RANKID % ${num_gpus} ))
gpu_id=$((PMIX_RANK % ${num_gpus} ))
#echo "The PMIX_RANK is ${PMIX_RANK}"
#echo "The PALS_LOCAL_RANKID is ${PALS_LOCAL_RANKID}"
#echo Local rank $PALS_LOCAL_RANKID running on gpu $gpu_id
#echo Local rank $PMIX_RANK running on gpu $gpu_id
export ZE_AFFINITY_MASK=$gpu_id
exec "$@"
