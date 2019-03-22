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

# Model
latent_dim=5

# Training
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
    $expdir/$db/datasets/$feaname/${dataset}.pkl


# Monophone system training.
steps/monophone.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm.yml \
    data/$db/lang \
    data/$db/$dataset \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/monophone_${feaname}


# Subspace HMM monophone training.
steps/subspace_monophone.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    --latent-dim $latent_dim \
    conf/hmm.yml \
    $expdir/$db/monophone_${feaname} \
    data/$db/$dataset \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/subspace_monophone_${feaname}_ldim${latent_dim}


# Subspace HMM monophone training.
steps/subspace_monophone.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    --latent-dim $latent_dim \
    --classes data/$db/lang/classes \
    conf/hmm.yml \
    $expdir/$db/monophone_${feaname} \
    data/$db/$dataset \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/dsubspace_monophone_${feaname}_ldim${latent_dim}

