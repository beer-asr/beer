#!/bin/bash

# Split a list into N equal chunks.


if [ $# -ne 3 ]; then
    echo "$0 <list> <n-split> <split-dir>"
    exit 1
fi

list=$1
n_split=$2
split_dir=$3
mkdir -p $split_dir

cp $list $split_dir/list
pushd $split_dir > /dev/null
split --numeric-suffixes=1 -n l/$n_split ./list
rm list
popd > /dev/null

