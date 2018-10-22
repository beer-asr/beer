#!/usr/bin/env bash

datadir=data
expdir=exp

mkdir -p $datadir $expdir


# Download the data and prepare the WAV files.
#local/prepare_mboshi_data.sh $datadir || exit 1


# Features extraction.
#mkdir -p features/mfcc
#beer features extract conf/mfcc.yml $datadir/train/wavs.scp features/mfcc || exit 1
#beer features archive features/mfcc features/mfcc.npz
#rm -fr features/mfcc # we don't need the original features anymore.

# Create the dataset.
mkdir -p $expdir/datasets
beer dataset create $datadir/train features/mfcc.npz \
    $expdir/datasets/mboshi_train_mfcc.pkl
