#!/bin/bash



if [ $# -ne 2 ]; then
    echo "$0 <setup.sh> <data/dataset>"
    exit 1
fi

setup=$1
. $setup
datadir=$2
scp="$datadir"/wav.scp
logdir="$datadir"/log
mkdir -p "$logdir"


# Check if we got the configuration and the scp files.
[[ -f "$fea_conf" ]] || { echo "$0: \"$fea_conf\" not found."; exit 1; }
[[ -f "$scp" ]] || { echo "$0: \"$scp\" not found."; exit 1; }

# Split the scp file into chunks to parallelize the features
# extraction.
mkdir -p "$datadir"/split
pushd "$datadir"/split > /dev/null
cp ../wav.scp ./
split --numeric-suffixes=1 -n l/$fea_njobs ./wav.scp
popd > /dev/null

# Cleanup the log files.
rm -f "$logdir"/extract-features.out.*

tmpdir=$(mktemp -d "$datadir"/beer.XXXX);
trap 'rm -rf "$tmpdir"' EXIT

# Extract the features on the SGE.
cmd="python utils/extract-features.py $fea_conf $tmpdir"
qsub \
    -N "beer-extract-features" \
    -cwd \
    -j y \
    -o "$logdir"/extract-features.out.'$TASK_ID' \
    -t 1-$fea_njobs \
    -sync y \
    $fea_sge_opts \
    utils/jobarray.qsub "$cmd" "$datadir"/split || exit 1

# Create the "npz" archives.
find "$tmpdir" -name '*npy' | zip -j -@ "$datadir"/feats.npz > /dev/null

