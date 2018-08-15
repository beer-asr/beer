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
        cp $datadir/local/data/$s.text $datadir/$s/text
        # Feature extraction
        python utils/prepare_trans.py \
            $datadir/$s/text $langdir/phones_48.txt $datadir/$s
    done
fi

if [ $stage -le 2 ]; then
    echo "Accumulating training data stastics"
    python utils/accumulate_data_stats.py \
        $datadir/train/feats.npz $datadir/train/feats.stats.npz
fi

if [ $stage -le 3 ]; then
    echo "Convert transcriptio into state sequences"
        python utils/prepare_lables.py \
            $langdir/phones_48.txt $datadir/train/text $nstate_per_phone
    echo "Initialize emission models"
    # To be done
fi



