#!/bin/sh


# Load the configuration.
setup="./setup.sh"
. $setup
stage=0

if [ $stage -le 0 ]; then
    echo =========================================================================
    echo "                         Data Preparation                              "
    echo =========================================================================
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

if [ $stage -le 1 ]; then
    echo =========================================================================
    echo "                         Features Extraction                           "
    echo =========================================================================
    for s in train test dev; do
        echo "Extracting features for: $s"
        steps/extract_features.sh $setup $datadir/$s || exit 1
    done

    # We need the mean/variance of the training data for the
    # initialization of the models. Also, we need to know how many
    # frames are in the training data to properly compute the
    # Stochastic Variational Lower Bound.
    echo "Computing training data statistics"
    python utils/compute_data_stats.py \
        $datadir/train/feats.npz $datadir/train/feats.stats.npz
fi

if [ $stage -le 3 ]; then
    echo "Convert the transcription into state sequences"
        python utils/prepare_labels.py \
            $langdir/phones.txt $datadir/train/text $nstate_per_phone
    echo "Initialize emission models"
    # To be done
fi

