#!/usr/bin/env bash

# Exit if one command fails.
set -e

########################################################################

## DIRECTORY STRUCTURE
datadir=data
feadir=features
expdir=exp

## DATA
db=timit
train_dataset=train
eval_dataset=test

## FEATURES
feaname=mfcc

## AUD
prior=dirichlet_process # Type of prior over the weights.
ngauss=4        # Number of Gaussian per state.
nunits=100      # maximum number of discovered units
epochs=40       # number of training epochs

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_61_to_39.txt"

########################################################################

# Load the BEER anaconda environment.
. path.sh

modeldir=$1


for epoch in $(seq 0 1 40); do
    for x in $train_dataset $eval_dataset; do
        #modeldir=$expdir/$db/aud_${feaname}_${ngauss}g_${prior}
        outdir=$modeldir/decode_perframe_e${epoch}/$x

        echo "--> Decoding $db/$x dataset"
        steps/decode.sh \
            --per-frame \
            --parallel-opts "-l mem_free=1G,ram_free=1G" \
            --parallel-njobs 30 \
            $modeldir/${epoch}.mdl \
            data/$db/$x \
            $expdir/$db/datasets/$feaname/${x}.pkl \
            $outdir

        if [ ! $x == "$train_dataset" ]; then
            au_mapping="--au-mapping $modeldir/decode_perframe_e${epoch}/$train_dataset/score/au_phone"
        else
            au_mapping=""
        fi

        echo "--> Scoring $db/$x dataset"
        steps/score_aud.sh \
            $au_mapping \
            $mapping \
            data/$db/$x/ali \
            $outdir/trans \
            $outdir/score
    done
done


