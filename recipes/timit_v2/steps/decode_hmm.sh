#!/bin/bash

if [ $# -ne 4 ];then
<<<<<<< HEAD
    echo "$0 <setup.sh> <model-dir> <date-test-dir> <decode-dir> "
=======
    echo "$0 <setup.sh> <date-test-dir> <decode-model> <decode-dir>"
>>>>>>> timit_recipe
    exit 1
fi

setup=$1
<<<<<<< HEAD
mdldir=$2
data_test_dir=$3
=======
data_test_dir=$2
mdl=$3
>>>>>>> timit_recipe
decode_dir=$4
stage=0

[ -f $setup ] && . $setup
<<<<<<< HEAD
=======
mdldir=$(dirname $mdl)
emission_conf=$mdldir/emissions.yml
>>>>>>> timit_recipe

mkdir -p $decode_dir/log

mdl=$mdldir/final.mdl
pdf_mapping=$mdldir/pdf_mapping.txt
for f in $mdl $pdf_mapping ; do
    [ ! -f $f ] && echo "No such file: $f" && exit 1;
done

if [ ! -f $decode_dir/decode_phone_ids.npz ];then
    echo "Decoding"
    python utils/decode_hmm.py $mdl $data_test_dir/feats.npz | \
        python utils/pdf2unit.py --phone-level $pdf_mapping | \
        python utils/prepare_trans.py $langdir/phones.txt \
        $decode_dir/decode_phone_ids.npz
fi

<<<<<<< HEAD
=======
if [ $stage -le 1 ];then
    echo "Merging states into phones"
    python utils/convert_states_to_phone.py \
        $decode_dir/decode_states.txt \
        $decode_dir/decode_results.txt \
        $langdir/phones.txt \
        $emission_conf > $decode_dir/log/convert_state_to_phone.log 2>&1 || exit 1
fi
>>>>>>> timit_recipe

if [ ! -f $decode_dir/decode_result.txt ];then
    python utils/score.py \
<<<<<<< HEAD
        $data_test_dir/phones.int.npz \
        $decode_dir/decode_phone_ids.npz \
        > $decode_dir/decode_result.txt || exit 1
=======
        --remove=$remove_sym \
        --duplicate=$duplicate \
        --phone_map=$phone_48_to_39_map \
        $data_test_dir/trans \
        $decode_dir/decode_results.txt > $decode_dir/log/score.log 2>&1 || exit 1
>>>>>>> timit_recipe
fi

cat $decode_dir/decode_result.txt

