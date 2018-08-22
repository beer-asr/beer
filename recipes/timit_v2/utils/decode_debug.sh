#!/bin/bash

if [ $# -ne 3 ];then
    echo "$0: <setup> <data-test-dir> <decode-modle-dir>"
    exit 1
fi

setup=$1
data_test_dir=$2
mdl_dir=$3
decode_dir=$mdl_dir/decode_test
[ -f $mdl_dir/decode_all_result.txt ] && rm $mdl_dir/decode_all_result.txt
touch $mdl_dir/decode_all_result.txt

for n in $(seq 1 29); do
    echo "Decoding with model $n"
    mdl=$mdl_dir/$n.mdl
    dir=${decode_dir}_${n}
    steps/decode_vae_hmm.sh $setup $data_test_dir $mdl $dir
    cut -d " " -f 5 $dir/decode_result.txt >> $mdl_dir/decode_all_result.txt
done
