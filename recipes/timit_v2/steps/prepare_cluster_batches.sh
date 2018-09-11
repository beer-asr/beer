#!/bin/bash

# Create a new dataset by stacking features.


if [ $# -ne 7 ]; then
    echo "$0 <setup.sh> <uttids> <fea.npz> <n_batches> <langdir> <alidir> <outdir>"
    exit 1
fi

setup=$1
uttids=$2
data=$3
n_batches=$4
langdir=$5
alidir=$6
outdir=$7

. $setup

mkdir -p $outdir

if [ ! -f $outdir/.done ]; then
    cat $uttids | shuf > $outdir/list.txt

    cmd="python utils/features-extract-central-frame.py \
            $langdir/phones.txt \
            $alidir/alis.npz \
            $alidir/pdf_mapping.txt \
            $data \
            $outdir/batchJOBID.npz"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "extract-central-frame" \
        "$fea_parallel_opts" \
        "$n_batches" \
        "$outdir/list.txt" \
        "$cmd" \
        "$outdir"
else
    echo "Batches already created. Skipping."
fi

