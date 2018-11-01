#!/usr/bin/env bash

feaname=mfcc
datadir=data
feadir=features
expdir=exp

mkdir -p $datadir $expdir $feadir


echo "--> Preparing data"
local/prepare_mboshi_data.sh $datadir || exit 1


echo "--> Extracting features"
for x in train dev; do
    steps/extract_features.sh conf/${feaname}.yml data/${x} $feadir/${x}
done

echo "--> Creating dataset(s)"
for x in train dev; do
    if [ ! -f $expdir/datasets/${x}.pkl ]; then
        echo "Creating \"${x}\" dataset."
        mkdir -p $expdir/datasets/$x

        # Create a "dataset". This "dataset" is just an object
        # associating the features with their utterance id and some
        # other meta-data (e.g. global mean, variance, ...).
        beer dataset create $datadir/$x $feadir/$x/mfcc.npz \
            $expdir/datasets/${x}.pkl
    else
        echo "Dataset \"${x}\" already created. Skipping."
    fi
done

