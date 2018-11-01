#!/usr/bin/env bash

databases="mboshi"
feaname=mfcc
datadir=data
feadir=features
expdir=exp

mkdir -p $datadir $expdir $feadir


for db in $databases; do
    echo "--> Preparing data for the $db database"
    local/$db/prepare_data.sh $datadir/$db || exit 1
done


for db in $databases; do
    echo "--> Extracting features for the $db database"
    for x in train dev; do
        steps/extract_features.sh conf/${feaname}.yml data/$db/$x \
            $feadir/$db/$x || exit 1
    done
done

# Create a "dataset". This "dataset" is just an object
# associating the features with their utterance id and some
# other meta-data (e.g. global mean, variance, ...).
for db in $databases; do
    echo "--> Creating dataset(s) for $db database"
    for x in train dev; do
        steps/create_dataset.sh data/$db/$x $feadir/$db/$x/${feaname}.npz \
            $expdir/$db/datasets/${x}.pkl
    done
done

