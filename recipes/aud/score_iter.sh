#!/usr/bin/env bash

# Exit if one command fails.
set -e

########################################################################

## DIRECTORY STRUCTURE
datadir=data
feadir=features
expdir=exp_dp

## DATA
db=mboshi
train=full

## FEATURES
feaname=mfcc

## AUD
prior=gamma_dirichlet_process # Type of prior over the weights.
ngauss=4        # Number of Gaussian per state.
nunits=100      # maximum number of discovered units
epochs=20       # number of training epochs

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_61_to_39.txt"

########################################################################

# Load the BEER anaconda environment.
. path.sh

modeldir=$1


for epoch in $(seq 0 1 20); do
    outdir=$modeldir/decode_perframe_e${epoch}/$train

    echo "--> Decoding $db/$train dataset"
    steps/decode.sh \
        --per-frame \
        --parallel-opts "-l mem_free=1G,ram_free=1G" \
        --parallel-njobs 30 \
        $modeldir/${epoch}.mdl \
        data/$db/$train \
        $expdir/$db/datasets/$feaname/${train}.pkl \
        $outdir

    echo "--> Scoring $db/$train dataset"
    steps/score_aud.sh \
        data/$db/$train/ali \
        $outdir/trans \
        $outdir/score
done


