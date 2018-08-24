#!/bin/bash

# Extract the features of a data set.


if [ $# -ne 2 ]; then
    echo "$0 <setup.sh> <data/dataset>"
    exit 1
fi

setup=$1
. $setup
datadir=$2
scp=$datadir/wav.scp


[[ -f $fea_conf ]] || { echo "File not found: $fea_conf" >2; exit 1; }
[[ -f $scp ]] || { echo "File not found: $fea_conf" >2; exit 1; }

# If the features are already created, do nothing.
if [ ! -f "$datadir"/feats.npz ]; then

    # Copy the configuration file to keep track of what kind of
    # features were extracted.
    cp $fea_conf $datadir/

    tmpdir=$(mktemp -d "$datadir"/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "beer-extract-features" \
        "$fea_parallel_opts" \
        $fea_njobs \
        $scp \
        "python utils/features-extract-parallel.py $fea_conf $tmpdir" \
        $datadir || exit 1

    # Create the "npz" archives.
    find "$tmpdir" -name '*npy' | zip -j -@ "$datadir"/feats.npz > /dev/null
else
    echo "Features already extracted in: $datadir/feats.npz."
fi


if [ ! -f "$datadir"/feats.stats.npz ]; then
    python utils/features-stats.py $datadir/feats.npz \
        $datadir/feats.stats.npz
else
   echo "Features statistics already computed in: $datadir/feats.stats.npz"
fi

