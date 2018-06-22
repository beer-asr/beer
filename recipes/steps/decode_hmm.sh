#!/bin/bash

if [ $# != 1 ]; then
    echo "$0: decode_setup.sh"
    exit 1
fi

setup=$1

. $setup

if [ ! -d $decode_dir ]; then
    mkdir -p $decode_dir || exit 1
fi
cp $setup $decode_dir

python3 steps/decode_hmm.py $model $decode_dir $feats \
                            $trans $phonelist $nstate_per_phone \
                            --gamma $gamma \
                            --phone_39 $phone_39 \
                            --remove_sys $remove_sys \
                            --score \
#                            --use-gpu
