#!/bin/bash


if [ $# -ne 1 ];then
    echo "$0: <setup.sh>"
    exit 1
fi

setup=$1
. $setup
stage=0

if [ $stage -le 0 ];then
    echo "Convert the transcription into state sequences"
        python utils/prepare_state_labels.py \
            $langdir/phones.txt $data_train_dir/phones.int.npz \
            $hmm_emission_conf $hmm_gmm_mdl_dir
fi

if [ $stage -le 1 ];then
    echo "Initialize emission models"
    python utils/create_emission.py \
        --stats $data_train_dir/feats.stats.npz \
        $hmm_emission_conf $hmm_gmm_mdl_dir/emission.mdl
fi

if [ $stage -le 2 ];then
    echo "Training HMM-GMM model"
    python -u -m cProfile -s cumtime utils/train_hmm.py \
        --infer_type $hmm_infer_type \
        --lrate $hmm_lrate \
        --batch_size $hmm_batch_size \
        --epochs $hmm_epochs \
        $data_train_dir/feats.npz $hmm_gmm_mdl_dir/states.int.npz \
        $hmm_gmm_mdl_dir/emission.mdl $data_train_dir/feats.stats.npz \
        $hmm_gmm_mdl_dir $use_gpu $hmm_fast_eval \
        > $hmm_gmm_mdl_dir/train.log 2>&1
fi
