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
feaname=mbn_babel

# AUD
nunits=100      # maximum number of discovered units
epochs=10       # number of training epochs

#######################################################################

# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db

echo "--> Preparing pseudo-phones \"language\" information"
mkdir -p data/$db/lang_aud

# The option "non-speech-unit" will force the decoder to start and end
# each utterance by a specific acoustic unit named "sil". The
# unit can also be freely decoded within the utterance. If your data
# is not well segmented and you are not sure that most of your data
# start and end with non-speech sound you better remove this option.
python utils/prepare_lang_aud.py \
    --non-speech-unit \
    $nunits > data/$db/lang_aud/units


echo "--> Extracting features for the $db database"
steps/extract_features.sh conf/${feaname}.yml $datadir/$db/$dataset \
     $feadir/$db/$dataset


# Create a "dataset". This "dataset" is just an object
# associating the features with their utterance id and some
# other meta-data (e.g. global mean, variance, ...).
echo "--> Creating dataset(s) for $db database"
steps/create_dataset.sh $datadir/$db/$dataset \
    $feadir/$db/$dataset/${feaname}.npz \
    $expdir/$db/datasets/$feaname/${dataset}.pkl


# AUD system training. You need to have a Sun Grid Engine like cluster
# (i.e. qsub command) to run it. If you have a different
# enviroment please see utils/parallel/sge/* to see how to adapt
# this recipe to you system.
steps/aud.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm.yml \
    data/$db/lang_aud \
    data/$db/$dataset/uttids \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/aud

