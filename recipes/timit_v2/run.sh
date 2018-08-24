#!/bin/bash


# Load the configuration.
if [ $# -ne 1 ]; then echo "$0 <setup.sh>"
    exit 1
fi


setup=$(pwd)/$1
. $setup || { echo "Could not load the $setup file." ; exit 1; }



# Set the stage you want to start from. Keep in mind that some steps
# depends on previous ones !!
stage=0


# Data preparation. Organize the data directory as:
#
#   data/
#     lang/
#       files related to the "language" (mostly phonetic information).
#     dataset/
#       files related to the dataset (features, transcription, ...)
#
if [ $stage -le 0 ]; then
    echo "--> Data preparation"
    local/timit_data_prep.sh "$timit" "$langdir" "$confdir" || exit 1
    for s in train test dev; do
        echo "Preparing for $datadir/$s"
        mkdir -p $datadir/$s
        cp $datadir/local/data/${s}_wav.scp $datadir/$s/wav.scp
        cp $datadir/local/data/$s.uttids $datadir/$s/uttids
        cp $datadir/local/data/$s.text $datadir/$s/trans
        cat $datadir/$s/trans | python utils/prepare_trans.py \
            $langdir/phones.txt $datadir/$s/phones.int.npz
    done
fi


# Extract the featuures for each dataset.
# Features for each data set will be store in "data/setname/feats.npz".
if [ $stage -le 1 ]; then
    echo "--> Features extraction"
    for s in train test dev; do
        echo "Extracting features for: $s"
        steps/extract_features.sh $setup $datadir/$s || exit 1
    done
fi


# HMM-GMM monophone.
if [ $stage -le 2 ]; then
    echo "--> HMM-GMM system"
    steps/train_hmm2.sh $setup $datadir/train $hmm_dir || exit 1
    steps/decode_hmm.sh $setup test_hmm_gmm $datadir/test \
        $hmm_dir/decode || exit 1
fi

