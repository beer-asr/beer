#!/usr/bin/env bash

if [ $# -ne 3 ]; then
    echo "usage: <data-dir> <fea-dir> <out-dataset>"
    exit 1
fi

datadir=$1
features=$2
output=$3
outdir=$(dirname $output)
mkdir -p $outdir

if [ ! -f $output ]; then
    beer dataset create $datadir $features $output
else
    echo "Dataset \"${x}\" already created. Skipping."
fi

