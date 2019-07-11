#!/usr/bin/env bash

# Exit if one command fails.
set -e

########################################################################
## SETUP

## DIRECTORY STRUCTURE
datadir=data
feadir=/mnt/scratch04/tmp/iondel/features
expdir=exp_ch1_v5

## DATA
db=timit
train=train
test=test

## FEATURES
feaname=mfcc

## AUD MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
ngauss=4        # number of Gaussian per state.
nunits=100      # maximum number of discovered units
epochs=30       # number of training epochs

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_61_to_39.txt"

########################################################################


# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db


echo "--> Preparing pseudo-phones \"language\" information"
mkdir -p data/$db/lang_aud

# The option "--non-speech-unit" will force the decoder to start and end
# each utterance by a specific acoustic unit named "sil". The
# unit can also be freely decoded within the utterance. If your data
# is not well segmented and you are not sure that most of your data
# start and end with non-speech sound it's safer to remove this option.
python utils/prepare_lang_aud.py \
    --non-speech-unit \
    $nunits > data/$db/lang_aud/units


for x in $train $test; do
    echo "--> Extracting features for the $db/$x database"
    steps/extract_features.sh conf/${feaname}.yml $datadir/$db/$x \
         $feadir/$db/$x

    # Create a "dataset". This "dataset" is just an object
    # associating the features with their utterance id and some
    # other meta-data (e.g. global mean, variance, ...).
    echo "--> Creating dataset(s) for $db database"
    steps/create_dataset.sh $datadir/$db/$x \
        $feadir/$db/$x/${feaname}.npz \
        $expdir/$db/datasets/$feaname/${x}.pkl
done


# AUD system training. You need to have a Sun Grid Engine like cluster
# (i.e. qsub command) to run it. If you have a different
# enviroment please see utils/parallel/sge/* to see how to adapt
# this recipe to you system.
steps/aud.sh \
    --prior $prior \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm_${ngauss}g.yml \
    data/$db/lang_aud \
    data/$db/$train \
    $expdir/$db/datasets/$feaname/${train}.pkl \
    $epochs $expdir/$db/$subset/aud_${feaname}_${ngauss}g_${prior}


for x in $train $test; do
    outdir=$expdir/$db/$subset/aud_${feaname}_${ngauss}g_${prior}/decode_perframe/$x

    echo "--> Decoding $db/$x dataset"
    steps/decode.sh \
        --per-frame \
        --parallel-opts "-l mem_free=1G,ram_free=1G" \
        --parallel-njobs 30 \
        $expdir/$db/$subset/aud_${feaname}_${ngauss}g_${prior}/final.mdl \
        data/$db/$subset/$x \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl \
        $outdir

    if [ ! $x == "$train" ]; then
        au_mapping="--au-mapping $expdir/$db/$subset/aud_${feaname}_${ngauss}g_${prior}/decode_perframe/$train/score/au_phone"
    fi

    echo "--> Scoring $db/$x dataset"
    steps/score_aud.sh \
        $au_mapping \
        $mapping \
        data/$db/$subset/$x/ali \
        $outdir/trans \
        $outdir/score
done

