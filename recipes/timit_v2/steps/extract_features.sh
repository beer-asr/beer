#!/bin/bash

# Extract the features of a data set.


if [ $# -ne 3 ]; then
    echo "$0 <setup.sh> <data/dataset> <fea-type>"
    exit 1
fi

setup=$1
. $setup
datadir=$2
fea_type=$3
scp=$datadir/wav.scp
fea_conf=$confdir/${fea_type}.yml


[[ -f $fea_conf ]] || { echo "File not found: $fea_conf" >2; exit 1; }
[[ -f $scp ]] || { echo "File not found: $fea_conf" >2; exit 1; }

# If the features are already created, do nothing.
if [ ! -f "$datadir"/${fea_type}.npz ]; then

    # Copy the configuration file to keep track of what kind of
    # features were extracted.
    cp $fea_conf $datadir/

    tmpdir=$(mktemp -d "$datadir"/beer.XXXX);
    trap 'rm -rf "$tmpdir"' EXIT
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "extract-$fea_type-features" \
        "$fea_parallel_opts" \
        $fea_njobs \
        $scp \
        "python utils/features-extract-parallel.py $fea_conf $tmpdir" \
        $datadir || exit 1

    # Create the "npz" archives.
    find "$tmpdir" -name '*npy' | zip -j -@ \
        "$datadir"/${fea_type}.npz > /dev/null
else
    echo "Features already extracted in: $datadir/${fea_type}.npz."
fi


if [ ! -f "$datadir"/${fea_type}.stats.npz ]; then
    python utils/features-stats.py $datadir/${fea_type}.npz\
        $datadir/${fea_type}.stats.npz
else
   echo "Features statistics already computed in: $datadir/${fea_type}.stats.npz"
fi

