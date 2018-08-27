#!/bin/bash

if [ $# -ne 4 ];then
    echo "$0: <setup> <data-test-dir> <decode-modle-dir> <hmm-or-vae>"
    exit 1
fi

setup=$1
. $setup
data_test_dir=$2
mdl_dir=$3
decode_type=$4
num_mdl=29
decode_dir=$mdl_dir/decode_test
pdf_mapping=$mdl_dir/pdf_mapping.txt
[ -f $mdl_dir/decode_all_result.txt ] && rm $mdl_dir/decode_all_result.txt
touch $mdl_dir/decode_all_result.txt

for n in $(seq 1 $num_mdl); do
    mdl=$mdl_dir/$n.mdl
    dir=${decode_dir}_${n}
    mkdir -p $dir/log
    if [ ! -f $dir/decode_results.txt ]; then
        echo "Decoding with model $mdl"
        if [ $decode_type == "hmm" ]; then
            python utils/decode_hmm.py $mdl $data_test_dir/feats.npz | \
                python utils/pdf2unit.py --phone-level $pdf_mapping \
                > $dir/decode_results.txt
        elif [ $decode_type == "vae-hmm" ]; then
            steps/decode_vae_hmm.sh $setup $data_test_dir $mdl $dir
        else
            echo "Wrong decode type: hmm or vae"
            exit 1
        fi
    fi
    echo "Scoring with model $mdl"
    python utils/score.py \
        --remove=$remove_sym \
        --duplicate=$duplicate \
        --phone_map=$phone_48_to_39_map \
        $data_test_dir/trans \
        $dir/decode_results.txt > $dir/log/score.log
    cat $dir/log/score.log >> $mdl_dir/decode_all_result.txt
done
