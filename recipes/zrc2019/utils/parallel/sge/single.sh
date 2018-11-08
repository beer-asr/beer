#!/bin/bash

# Submit a job on the SGE.


if [ $# -ne 4 ]; then
    echo "$0 <name> <opts> <cmd> <log-dir>"
    exit 1
fi

name=$1
opts=$2
cmd=$3
log_dir=$4

rm -fr $log_dir/${name}.out
qsub -N "$name"  -cwd  -j y -sync y\
    -o $log_dir/${name}.out \
    $opts utils/parallel/sge/job.qsub "$cmd" || exit 1

