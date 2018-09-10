#!/bin/bash

# Submit an array job on the SGE.


if [ $# -ne 5 ]; then
    echo "$0 <name> <opts> <split-dir> <cmd> <log-dir>"
    exit 1
fi

name=$1
opts=$2
split_dir=$3
cmd=$4
log_dir=$5

cmd=$(echo $cmd | sed s/%JOBID/$TASK_ID/g)
rm -fr $log_dir/${name}.out.*
qsub -N "$name"  -cwd  -j y -sync y\
    -o $log_dir/${name}.out'.$TASK_ID' \
    -t 1-$(ls $split_dir | wc -l) \
    $opts utils/parallel/sge/jobarray.qsub "$cmd" $split_dir || exit 1

