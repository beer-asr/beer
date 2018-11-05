#!/usr/bin/env bash

#######################################################################
## SETUP

# Directory structure
datadir=data
feadir=features
expdir=exp

# Data
db=mboshi
dataset=train

# Features
feaname=mfcc

# AUD training
epochs=5
lrate=0.1
batch_size=400

#######################################################################

source activate beer

mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db || exit 1



echo "--> Extracting features for the $db database"
steps/extract_features.sh conf/${feaname}.yml $datadir/$db/$dataset \
     $feadir/$db/$dataset || exit 1


# Create a "dataset". This "dataset" is just an object
# associating the features with their utterance id and some
# other meta-data (e.g. global mean, variance, ...).
echo "--> Creating dataset(s) for $db database"
steps/create_dataset.sh $datadir/$db/$dataset \
    $feadir/$db/$dataset/${feaname}.npz \
    $expdir/$db/datasets/${dataset}.pkl


echo "--> Acoustic Unit Discovery on $db database"
steps/aud.sh conf/hmm.yml $expdir/$db/datasets/${dataset}.pkl \
    $epochs $lrate $batch_size $expdir/$db/aud

