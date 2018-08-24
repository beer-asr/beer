#!/bin/bash

if [ $# -ne 3 ];then
    echo "$0: <setup> <data-test-dir> <decode-modle-dir>"
    exit 1
fi

setup=$1
data_test_dir=$2
mdl_dir=$3
decode_dir=$mdl_dir/decode_test
mkdir -p $decode_dir
[ -f $mdl_dir/decode_all_result.txt ] && rm $mdl_dir/decode_all_result.txt
touch $mdl_dir/decode_all_result.txt

for n in $(seq 1 5); do
    echo "Decoding with model $n"
    mdl=$mdl_dir/$n.mdl
    dir=${decode_dir}_${n}
    echo $mdl
    steps/decode_vae_hmm.sh $setup $mdl $data_test_dir $dir
done
