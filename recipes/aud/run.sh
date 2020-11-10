#!/usr/bin/env bash

# Exit if one command fails.
set -e

########################################################################
## SETUP

## DIRECTORY STRUCTURE
datadir=data
feadir=/mnt/scratch04/tmp/iondel/features
expdir=exp

## DATA
db=timit
train=full
test=full

## FEATURES
feaname=mfcc

## AUD MODEL

# Type of prior over the weights.
# Possible choices are:
#   - dirichlet                        -> Dirichlet distribution
#   - dirichlet_process                -> Dirichlet process
#   - gamma_dirichlet_process          -> Dirichlet process with Gamma prior
prior=gamma_dirichlet_process

nunits=100      # maximum number of discovered units
epochs=30       # number of training epochs

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


echo "--> Extracting features for the $db/$train database"
steps/extract_features.sh \
    conf/${feaname}.yml \
    $datadir/$db/$train/wav.scp \
    $feadir/$db/$train

# Create a "dataset". This "dataset" is just an object
# associating the features with their utterance id and some
# other meta-data (e.g. global mean, variance, ...).
echo "--> Creating dataset(s) for $db database"
steps/create_dataset.sh $datadir/$db/$train \
    $feadir/$db/$train/${feaname}.npz \
    $expdir/$db/datasets/$feaname/${train}.pkl


# AUD system training. You need to have a Sun Grid Engine like cluster
# (i.e. qsub command) to run it. If you have a different
# enviroment please see utils/parallel/sge/* to see how to adapt
# this recipe to your system.
steps/aud.sh \
    --prior $prior \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm.yml \
    data/$db/lang_aud \
    data/$db/$train \
    $expdir/$db/datasets/$feaname/${train}.pkl \
    $epochs $expdir/$db/$subset/aud_${feaname}_${prior}

outdir=$expdir/$db/$subset/aud_${feaname}_${prior}/decode_perframe/$train

echo "--> Decoding $db/$train dataset"
steps/decode.sh \
    --per-frame \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    $expdir/$db/$subset/aud_${feaname}_${prior}/final.mdl \
    data/$db/$subset/$train/uttids \
    $expdir/$db/$subset/datasets/$feaname/${train}.pkl \
    $outdir

echo "--> Scoring $db/$train dataset"
steps/score_aud.sh \
    data/$db/$subset/$train/ali \
    $outdir/trans \
    $outdir/score

