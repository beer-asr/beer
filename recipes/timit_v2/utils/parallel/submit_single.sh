#!/bin/bash

# Sumbit a task on the the selected environment (local/sge).


if [ $# -ne 5 ]; then
    echo "$0: <parallel-env> <name> <opts> <cmd> <outdir>"
    exit 1
fi

parallel_env=$1
name=$2
opts=$3
cmd=$4
outdir=$5

# Submit the jobs
mkdir -p $outdir/log
utils/parallel/$parallel_env/single.sh $name "$opts" "$cmd" \
    $outdir/log 2>&1 > $outdir/parallel.out || exit 1

