#!/bin/bash

# Prepare the lang directory for the Acoustic Unit Discovery system.

if [ $# -ne 2 ]; then
    echo "Usage: $0 <n-units> <lang-dir>"
    exit 1
fi

n_units=$1
lang_dir=$2

mkdir -p $lang_dir

echo sil 0 > $lang_dir/phones.txt
for unit_id in $(seq 1 $((n_units))); do
    echo a$unit_id $unit_id >> $lang_dir/phones.txt
done
