#!/bin/bash

# Sumbit a set of tasks on the the selected environment (local/sge).


if [ $# -ne 7 ]; then
    echo "$0: <parallel-env> <name> <opts> <njobs> <list> <cmd> <outdir>"
    exit 1
fi

parallel_env=$1
name=$2
opts=$3
njobs=$4
list=$5
cmd=$6
outdir=$7

mkdir -p $outdir/{split,log}

# Split the list into "njobs" sub lists.
utils/parallel/split.sh $list $njobs $outdir/split

# Submit the jobs
utils/parallel/$parallel_env/parallel.sh $name "$opts" $outdir/split "$cmd" \
    $outdir/log 2>&1 > $outdir/log/parallel.log

