#!/bin/bash

help="\
Compute the statistics (mean/var/counts) of a database.

usage: $0 <archives-list> <out-npzfile>
"

if [ $# -ne 2 ]; then
    echo "${help}"
    exit 1
fi

archives=$1
outfile=$2

if [ ! -f "${outfile}" ]; then
    echo "Computing stats..."
    python utils/compute-stats.py ${archives} ${outfile} || exit 1
else
    echo "Statistics already computed. Skipping."
fi

