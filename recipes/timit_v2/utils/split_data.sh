#!/bin/bash

if [ $# -ne 2 ];then
    echo "$0: <srcdir> <num_job>"
    exit 1
fi
srcdir=$1
num_job=$2

for f in $srcdir/{"feats.npz","trans","phones.int.npz"};do
    [ ! -f $f ] && echo "File not found: $f" && exit 1;
done

mkdir -p $srcdir/split_${num_job}
python utils/split_data.py \
    $srcdir $srcdir/split_${num_job} \
    $num_job || exit 1


