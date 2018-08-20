#!/bin/bash

exit_msg (){
    echo "$1"
    exit 1
}

if [ $# -ne 3 ];then
    echo "$0: <setup.sh> <data-train-dir> <mdl-dir>"
    exit 1
fi

setup=$1
data_train_dir=$2
mdl_dir=$3

. $setup

[ -f "$hmm_emission_conf" ] || exit_msg "File not found: $hmm_emission_conf"

if [ ! -d $mdl_dir ];then
    mkdir -p $mdl_dir/log
    cp $setup $mdl_dir
    cp $fea_conf $mdl_dir
    cp $hmm_emission_conf $mdl_dir
fi

if [ ! -f $mdl_dir/states.int.npz ];then
    echo "Convert the transcription into state sequences"
        python utils/prepare_state_labels.py \
            $langdir/phones.txt \
            $data_train_dir/phones.int.npz \
            $hmm_emission_conf \
            $mdl_dir || exit_msg "Failed to create state labels"
else
    echo "Phone state labels already created in $mdl_dir/states.int.npz"
fi

if [ ! -f $mdl_dir/emission.mdl ];then
    echo "Initialize emission models"
    python utils/create_emission.py \
        --stats $data_train_dir/feats.stats.npz \
        $hmm_emission_conf $mdl_dir/emission.mdl
else
    echo "Emissions already created: $mdl_dir/emission.mdl"
fi

if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training HMM-GMM model"
    python -u -m cProfile -s cumtime utils/train_hmm.py \
        --infer_type $hmm_infer_type \
        --lrate $hmm_lrate \
        --batch_size $hmm_batch_size \
        --epochs $hmm_epochs \
        $data_train_dir/feats.npz $mdl_dir/states.int.npz \
        $mdl_dir/emission.mdl $data_train_dir/feats.stats.npz \
        $mdl_dir $use_gpu $hmm_fast_eval \
        > $mdl_dir/log/train.log 2>&1 || exit_msg "Training failed"
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi
