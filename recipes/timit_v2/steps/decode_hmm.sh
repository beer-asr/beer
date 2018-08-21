#!/bin/bash

if [ $# -ne 4 ];then
    echo "$0 <setup.sh> <date-test-dir> <decode-model> <decode-dir>"
    exit 1
fi

setup=$1
data_test_dir=$2
mdl=$3
decode_dir=$4
stage=0

[ -f $setup ] && . $setup
mdldir=$(dirname $mdl)
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
        $decode_dir/decode_results.txt \
        $langdir/phones.txt \
        $emission_conf > $decode_dir/log/convert_state_to_phone.log 2>&1 || exit 1
fi

if [ $stage -le 2 ];then
    echo "DTW scoring"
    python utils/score.py \
        --remove=$remove_sym \
        --duplicate=$duplicate \
        --phone_map=$phone_48_to_39_map \
        $data_test_dir/trans \
        $decode_dir/decode_results.txt > $decode_dir/log/score.log 2>&1 || exit 1
fi
