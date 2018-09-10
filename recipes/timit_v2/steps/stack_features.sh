#!/bin/bash

# Create a new dataset by stacking features.


if [ $# -ne 5 ]; then
    echo "$0 <setup.sh> <uttids> <dataset.npz> <context> <outdir>"
    exit 1
fi

setup=$1
uttids=$2
data=$3
context=$4
outdir=$5
mkdir -p $outdir

. $setup

fea_name=$(basename $data)
fea_name=${fea_name%.*}

if [ ! -f $outdir/${fea_name}_c${context}.npz ]; then
    tmpdir=$(mktemp -d $outdir/beer.tmp.XXX)
    trap 'rm -rf "$tmpdir"' EXIT

    cmd="python utils/features-stack-parallel.py \
            --context $context \
            $data $tmpdir"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "stack-features" \
        "$fea_parallel_opts" \
        "$fea_njobs" \
        $uttids \
        "$cmd" \
        "$outdir"

    # Create the "npz" archives.
    find "$tmpdir" -name '*npy' | zip -j -@ \
        "$outdir"/${fea_name}_c${context}.npz > /dev/null
else
    echo "Stacked features with context $context already created in: $outdir/${fea_name}_c${context}.npz"
fi

