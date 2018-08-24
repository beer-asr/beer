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
lists=$5
cmd=$6
outdir=$7
mkdir -p $outdir/{split,log}


# Split the list into "njobs" sub lists.
if [ ! -d $outdir/split ]; then
    utils/parallel/split.sh $list $njobs $outdir/split
fi

# Submit the jobs
echo utils/parallel/$parallel_env/parallel.sh $name \"$opts\" $outdir/split \"$cmd\" \
    $outdir/log
utils/parallel/$parallel_env/parallel.sh $name "$opts" $outdir/split "$cmd" \
    $outdir/log

