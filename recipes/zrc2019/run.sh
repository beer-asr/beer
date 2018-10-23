#!/usr/bin/env bash

datadir=data
feadir=features
expdir=exp

mkdir -p $datadir $expdir $feadir


echo "--> Preparing data"
local/prepare_mboshi_data.sh $datadir || exit 1


echo "--> Extracting features"
for x in train dev; do
    if [ ! -f $feadir/$x/mfcc.npz ]; then
        mkdir -p $feadir/$x/mfcc

        echo "Extracting features for the \"${x}\" dataset."

        # Extract the features.
        beer features extract conf/mfcc.yml $datadir/$x/wavs.scp \
            $feadir/$x/mfcc || exit 1

        # Put all the features files into a single archive.
        beer features archive $feadir/$x/mfcc $feadir/$x/mfcc.npz

        # We don't need the original features anymore as they are stored in
        # the archive.
        rm -fr $feadir/$x/mfcc
    else
        echo "Features already extracted for the \"${x}\" dataset. Skipping."
    fi
done


echo "--> Creating dataset(s)"
for x in train dev; do
    if [ ! -f $expdir/datasets/${x}.pkl ]; then
        echo "Creating \"${x}\" dataset."
        mkdir -p $expdir/datasets/$x

        # Create a "dataset". This "dataset" is just an object
        # associating the features with their utterance id and some
        # other meta-data (e.g. spk info if available).
        beer dataset create $datadir/$x $feadir/$x/mfcc.npz \
            $expdir/datasets/${x}.pkl
    else
        echo "Dataset \"${x}\" already created. Skipping."
    fi
done

