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
plang=GE_PO_SP
olddb=globalphone/$plang
db=timitfull
dataset=all

# Features
feaname=mfcc

# Model
latent_dim=100

# Training
ngauss=4        # Number of Gaussian per state.
epochs=30

#######################################################################

# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $expdir

echo "--> Training the subspace Bayesian AUD system"
steps/subspace_aud.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    --latent-dim $latent_dim \
    conf/hmm_${ngauss}g.yml \
    $expdir/$olddb/subspace_monophone_${feaname}_g${ngauss}_ldim${latent_dim}/gsm_30.mdl \
    $expdir/$db/aud_${feaname}_${ngauss}g \
    data/$db/$dataset \
    $expdir/$db/datasets/$feaname/${dataset}.pkl \
    $epochs $expdir/$db/subspace_aud_${feaname}_${ngauss}g_ldim${latent_dim}_pdata${plang}

