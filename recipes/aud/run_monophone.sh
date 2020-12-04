#!/usr/bin/env bash

# Exit if one command fails.
set -e

########################################################################
## SETUP

## DIRECTORY STRUCTURE
datadir=data        # where will stored the corpus specific data (transcription, dictionary, ...)
feadir=/mnt/scratch04/tmp/iondel/features     # where will be stored the features
expdir=exp          # experiment directory where will be stored the models and the results

## DATA
db=timit            # name of the corpus (timit, mboshi, globalphone)
#subset=             # subset of the corpus (mostly used for globalphone: FR, GE, ...)
train=train         # name of the train set (usually "train")
test=test           # name of the test set (usuall "test")

## FEATURES
feaname=mfcc

## MONOPHONE MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
latent_dim=100      # latent dimension of the subspace model
epochs=30           # number of training epochs

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_48_to_39.txt"

########################################################################


# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db $subset


for x in $train $test; do
    echo "--> Extracting features for the $db/$x database"
    steps/extract_features.sh \
        conf/${feaname}.yml \
        $datadir/$db/$subset/$x/wav.scp \
        $feadir/$db/$subset/$x

    # Create a "dataset". This "dataset" is just an object
    # associating the features with their utterance id and some
    # other meta-data (e.g. global mean, variance, ...).
    echo "--> Creating dataset(s) for $db database"
    steps/create_dataset.sh $datadir/$db/$subset/$x \
        $feadir/$db/$subset/$x/${feaname}.npz \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl
done


# Monophone system training.
steps/monophone.sh \
    --prior $prior \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm.yml \
    data/$db/$subset/lang \
    data/$db/$subset/$train/ \
    $expdir/$db/$subset/datasets/$feaname/${train}.pkl \
    $epochs \
    $expdir/$db/$subset/monophone_${feaname}_${prior}


# Subspace HMM monophone training.
steps/subspace_monophone.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    --latent-dim $latent_dim \
    conf/hmm.yml \
    $expdir/$db/$subset/monophone_${feaname}_${prior} \
    data/$db/$subset/$train/ \
    $expdir/$db/$subset/datasets/$feaname/${train}.pkl \
    $epochs \
    $expdir/$db/$subset/subspace_monophone_${feaname}_${prior}_ldim${latent_dim}


for x in $test; do
    outdir=$expdir/$db/$subset/subspace_monophone_${feaname}_${prior}_ldim${latent_dim}/decode_perframe/$x

    echo "--> Decoding $db/$x dataset"
    steps/decode.sh \
        --per-frame \
        --parallel-opts "-l mem_free=1G,ram_free=1G" \
        --parallel-njobs 30 \
        $expdir/$db/$subset/subspace_monophone_${feaname}_${prior}_ldim${latent_dim}/final.mdl \
        data/$db/$subset/$x/uttids \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl \
        $outdir

    echo "--> Scoring $db/$x dataset"
    steps/score_aud.sh \
        $au_mapping \
        $mapping \
        data/$db/$subset/$x/ali \
        $outdir/trans \
        $outdir/score
done

