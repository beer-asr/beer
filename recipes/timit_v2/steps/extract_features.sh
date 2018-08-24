#!/bin/bash

# Extract the features of a data set.


if [ $# -ne 2 ]; then
    echo "$0 <setup.sh> <data/dataset>"
    exit 1
fi

setup=$1
. $setup
datadir=$2
scp="$datadir"/wav.scp

# Check if we got the configuration and the scp files.
if [ ! -f "$fea_conf" ]; then
    echo "\"$fea_conf\" not found."
    exit 1
else
    cp $fea_conf $datadir
fi
if [ ! -f "$scp" ]; then
    echo "\"$scp\" not found."
    exit 1
fi

# If the features are already created, do nothing.
if [ ! -f "$datadir"/feats.npz ]; then

    tmpdir=$(mktemp -d "$datadir"/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT

    utils/parallel/submit_parallel.sh \
        $parallel_env \
        "beer-extract-features" \
        "$fea_parallel_opts" \
        $fea_njobs \
        $datadir/wav.scp \
        "python utils/extract-features.py $fea_conf $tmpdir" \
        $datadir || exit 1

    # Create the "npz" archives.
    find "$tmpdir" -name '*npy' | zip -j -@ "$datadir"/feats.npz > /dev/null
else
    echo "Features already extracted in: $datadir/feats.npz."
fi


if [ ! -f "$datadir"/feats.stats.npz ]; then
    python utils/compute_data_stats.py \
        $datadir/feats.npz $datadir/feats.stats.npz
else
   echo "Features statistics already computed in: $datadir/feats.stats.npz"
fi

