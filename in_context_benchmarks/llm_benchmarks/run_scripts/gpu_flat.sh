#!/bin/bash

$(which sh) -c 'ZE_AFFINITY_MASK=$((PALS_LOCAL_RANKID % 12)) exec "$@"'
