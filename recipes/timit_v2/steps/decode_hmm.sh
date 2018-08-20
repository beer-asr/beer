#!/bin/bash

if [ $# -ne 3 ];then
    echo "$0 <setup.sh> <date-test-dir> <decode-dir>"
    exit 1
fi

setup=$1
data_test_dir=$2
decode_dir=$3
stage=0

[ -f $setup ] && . $setup
mdldir=$(dirname $decode_dir)
mdl=$mdldir/final.mdl
emission_conf=$mdldir/emissions.yml

mkdir -p $decode_dir/log
for f in $mdl $emission_conf; do
    [ ! -f $f ] && echo "No such file: $f" && exit 1;
done
if [ $stage -le 0 ];then
    echo "Decoding"
    python utils/decode_hmm.py \
        --gamma $hmm_gamma \
        $mdl $decode_dir \
        $data_test_dir/feats.npz $emission_conf \
        > $decode_dir/log/decode.log 2>&1 || exit 1
fi

if [ $stage -le 1 ];then
    echo "Merging states into phones"
    python utils/convert_states_to_phone.py \
        $decode_dir/decode_states.txt \
        $decode_dir/decode_phone_ids.npz \
        $langdir/phones.txt \
        $emission_conf > $decode_dir/log/convert_state_to_phone.log 2>&1 || exit 1
fi

if [ $stage -le 2 ];then
    echo "DTW scoring"
    python utils/score.py \
        $data_test_dir/phones.int.npz \
        $decode_dir/decode_phone_ids.npz \
        $decode_dir/decode_result.txt > $decode_dir/log/score.log 2>&1 || exit 1
fi
