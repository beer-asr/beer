#!/bin/sh


# Load the configuration.
setup="./setup.sh"
. $setup
stage=1

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
