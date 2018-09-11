#!/bin/bash


if [ $# -ne 1 ];then
    echo "$0: <data-dir>"
    exit 1
fi

data_dir=$1

if [ ! -f $data_dir/stats.npz ]; then
    find $data_dir -name 'batch*npz' > $data_dir/batches || exit 1
    python utils/features-stats.py $data_dir/batches $data_dir/stats.npz || exit 1
else
    echo "Data already prepared. Skipping."
fi

