#!/usr/bin/env bash

# Exit if one command fails.
set -e

#######################################################################
## SETUP

# Directory structure
datadir=data
feadir=features
expdir=exp

# Data
db=timit
dataset=train

# Features
feaname=mfcc

# AUD training
# The number of epochs probably needs to be tuned to the final data.
epochs=30

#######################################################################

# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db


echo "--> Extracting features for the $db database"
steps/extract_features.sh conf/${feaname}.yml $datadir/$db/$dataset \
     $feadir/$db/$dataset


# Create a "dataset". This "dataset" is just an object
# associating the features with their utterance id and some
# other meta-data (e.g. global mean, variance, ...).
echo "--> Creating dataset(s) for $db database"
steps/create_dataset.sh $datadir/$db/$dataset \
    $feadir/$db/$dataset/${feaname}.npz \
    $expdir/$db/datasets/${dataset}.pkl


# AUD system training. You need to have a Sun Grid Engine like cluster
# (i.e. qsub command) to run it. If you have a different
# enviroment please see utils/parallel/sge/* to see how to adapt
# this recipe to you system.
steps/aud.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm.yml \
    data/$db/train/uttids \
    $expdir/$db/datasets/${dataset}.pkl \
    $epochs $expdir/$db/aud

