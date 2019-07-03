#!/usr/bin/env bash

# Exit if one command fails.
set -e

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <corpus> <gsm-init> [<subset>] "
    echo ""
    echo "Build a subspace AUD system"
    echo ""
    echo "Examples:"
    echo "  $ $0 timit exp/globalphone/FR/subspace_monophone/gsm_final.mdl"
    echo "  $ $0 globalphone exp/globalphone/FR/subspace_monophone/gsm_final.mdl GE"
    echo ""
    exit 1
fi

########################################################################
## SETUP

## DIRECTORY STRUCTURE
datadir=data        # where will stored the corpus specific data (transcription, dictionary, ...)
feadir=features     # where will be stored the features
expdir=exp          # experiment directory where will be stored the models and the results

## DATA
db=$1               # name of the corpus (timit, mboshi, globalphone)
subset=$3           # subset of the corpus (mostly used for globalphone: FR, GE, ...)
train=train         # name of the train set (usually "train")
test=test           # name of the test set (usuall "test")

## FEATURES
feaname=mfcc

## AUD MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
ngauss=4            # number of Gaussian per state.
nunits=100          # maximum number of discovered units
epochs=40           # number of training epochs
gsm_init=$2         # GSM for initialization
outdir=$expdir/$db/$subset/subspace_aud_${feaname}_${ngauss}g_${prior}

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_61_to_39.txt"

########################################################################


# Load the BEER anaconda environment.
. path.sh


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db $subset


echo "--> Preparing pseudo-phones \"language\" information"
mkdir -p data/$db/$subset/lang_aud

# The option "--non-speech-unit" will force the decoder to start and end
# each utterance by a specific acoustic unit named "sil". The
# unit can also be freely decoded within the utterance. If your data
# is not well segmented and you are not sure that most of your data
# start and end with non-speech sound it's safer to remove this option.
python utils/prepare_lang_aud.py \
    --non-speech-unit \
    $nunits > data/$db/$subset/lang_aud/units


for x in $train $test; do
    echo "--> Extracting features for the $db/$x database"
    steps/extract_features.sh \
        conf/${feaname}.yml \
        $datadir/$db/$subset/$x \
        $feadir/$db/$subset/$x

    # Create a "dataset". This "dataset" is just an object
    # associating the features with their utterance id and some
    # other meta-data (e.g. global mean, variance, ...).
    echo "--> Creating dataset(s) for $db database"
    steps/create_dataset.sh $datadir/$db/$subset/$x \
        $feadir/$db/$subset/$x/${feaname}.npz \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl
done


echo "--> Training the subspace Bayesian AUD system"
steps/subspace_aud.sh \
    --parallel-opts "-l mem_free=1G,ram_free=1G" \
    --parallel-njobs 30 \
    conf/hmm_${ngauss}g.yml \
    $gsm_init \
    data/$db/$subset/lang_aud \
    data/$db/$subset/$train \
    $expdir/$db/$subset/datasets/$feaname/${train}.pkl \
    $epochs $outdir


for x in $train $test; do
    score_outdir=$outdir/decode_perframe/$x

    echo "--> Decoding $db/$x dataset"
    steps/decode.sh \
        --per-frame \
        --parallel-opts "-l mem_free=2G,ram_free=2G" \
        --parallel-njobs 30 \
        $outdir/final.mdl \
        data/$db/$subset/$x \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl \
        $score_outdir

    if [ ! $x == "$train" ]; then
        au_mapping="--au-mapping $outdir/decode_perframe/$train/score/au_phone"
    fi

    echo "--> Scoring $db/$x dataset"
    steps/score_aud.sh \
        $au_mapping \
        $mapping \
        data/$db/$subset/$x/ali \
        $score_outdir/trans \
        $score_outdir/score
done

