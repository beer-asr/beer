#!/bin/bash

# Split a list of utterance ids into N equal chunks.


if [ $# -ne 2 ]; then
    echo "$0 <utt-list> <n-split> <split-dir>"
    exit 1
fi

utt_list=$1
n_split=$2
split_dir=$3
mkdir -p $split_dir

pushd $split_dir > /dev/null
split --numeric-suffixes=1 -n l/$n_split $utt_list
popd > /dev/null

