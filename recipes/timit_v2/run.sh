#!/bin/sh


# Load the configuration.

if [ $# -ne 2 ]; then
    echo "$0 <setup.sh> num_phones"
    exit 1
fi
setup=$(pwd)/$1
. $setup
num_phns=$2
stage=-1

if [ $stage -le 0 ]; then
    echo ======================================================================
    echo "                         Data Preparation                           "
    echo ======================================================================
    if [ $num_phns == "48" ]; then
        echo "Default: preparing data with 48 phonemes"
        local/timit_data_prep.sh "$timit" "$datadir" "$confdir" || exit 1
    elif [ $num_phns == "61" ]; then
        echo "Preparing data with 61 phonemes"
        local/timit_data_prep_61_phns.sh "$timit" "$datadir" "$confdir" || exit 1
    else
        echo "Wrong number of phonemes: 48 or 61 !"
        exit 1
    fi
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
if [ $stage -le 2 ]; then
    echo "---------- Features extraction ----------"
    for s in train test dev; do
        echo "Extracting features for: $s"
        steps/extract_features.sh $setup $datadir/$s || exit 1
    done
fi


# HMM-GMM monophone.
if [ $stage -le 3 ]; then
    echo "---------- HMM-GMM system ----------"
    steps/train_hmm2.sh $setup $datadir/train test_hmm_gmm || exit 1
    steps/decode_hmm.sh $setup test_hmm_gmm $datadir/test test_hmm_gmm/decode || exit 1
fi

