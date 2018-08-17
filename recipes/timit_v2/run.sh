#!/bin/sh


# Load the configuration.

if [ $# -ne 1 ]; then
    echo "$0 <setup.sh>"
    exit 1
fi
setup=$(pwd)/$1
. $setup
stage=0

if [ $stage -le 0 ]; then
    echo ======================================================================
    echo "                         Data Preparation                           "
    echo ======================================================================
    local/timit_data_prep.sh "$timit" "$langdir" "$confdir" || exit 1
fi

if [ $stage -le 1 ]; then
    for s in train test dev; do
        echo "Preparing for $datadir/$s"
        mkdir -p $datadir/$s
        cp $datadir/local/data/${s}_wav.scp $datadir/$s/wav.scp
        cp $datadir/local/data/$s.uttids $datadir/$s/uttids
        cp $datadir/local/data/$s.text $datadir/$s/trans
        python utils/prepare_trans.py \
            $datadir/$s/trans $langdir/phones.txt $datadir/$s
    done
fi

if [ $stage -le 2 ]; then
    echo ======================================================================
    echo "                         Features Extraction                        "
    echo ======================================================================
    for s in train test dev; do
        echo "Extracting features for: $s"
        steps/extract_features.sh $setup $datadir/$s || exit 1
    done
fi

if [ $stage -le 3 ]; then
    echo ======================================================================
    echo "                         HMM-GMM Training                           "
    echo ======================================================================
    echo "Convert the transcription into state sequences"
        python utils/prepare_labels.py \
            $langdir/phones.txt $data_train_dir/phones.int.npz \
            $hmm_conf $hmm_gmm_mdl_dir
    echo "Initialize emission models"
    python utils/create_emission.py \
        --stats $data_train_dir/feats.stats.npz \
        $hmm_conf $hmm_gmm_mdl_dir/emission.mdl
    echo "Training HMM-GMM model"
    python utils/train_hmm.py \
        --infer_type $hmm_infer_type \
        --lrate $hmm_lrate \
        --batch_size $hmm_batch_size \
        --epochs $hmm_epochs \
        $data_train_dir/feats.npz $hmm_gmm_mdl_dir/states.int.npz \
        $hmm_gmm_mdl_dir/emission.mdl $data_train_dir/feats.stats.npz \
        $hmm_gmm_mdl_dir $use_gpu $hmm_fast_eval \
        > $hmm_gmm_mdl_dir/train.log 2>&1
fi

