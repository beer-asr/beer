#!/usr/bin/env bash

# Exit if one command fails.
set -e


echo "$0 $@"
########################################################################
## SETUP

parallel_njobs=1000

decode_all=false
decode_freq=10
hsubspace_opts="--gsm-std-lrate 5e-3 --opts-conf grad_large.conf"
nosil=false
## DIRECTORY STRUCTURE
datadir=data        # where will stored the corpus specific data (transcription, dictionary, ...)
feadir=/mnt/scratch04/tmp/xyusuf00/features     # where will be stored the features
expdir=exp_aud          # experiment directory where will be stored the models and the results

## DATA
train=full         # name of the train set (usually "train")
test=full           # name of the test set (usually "test")

## FEATURES
feaname=mfcc

## AUD MODEL
prior=gamma_dirichlet_process # Type of prior over the weights.
ngauss=4            # number of Gaussian per state.
nunits=100          # maximum number of discovered units
epochs=100           # number of training epochs
concentration=1     # concentration parameter of the DP

## SCORING
# This option is mostly for TIMIT.
mapping="--mapping data/timit/lang/phones_61_to_39.txt"
mapping=""

########################################################################


[ -f ./kutils/parse_options.sh ] && . ./kutils/parse_options.sh
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <corpus> <gsm-init-dir> [<subset>] "
    echo ""
    echo "Build a hierarchical subspace AUD system"
    echo ""
    echo "Examples:"
    echo "  $ $0 timit exp/globalphone/FR/subspace_monophone/"
    echo "  $ $0 globalphone exp/globalphone/FR/subspace_monophone/ GE"
    echo ""
    exit 1
fi

db=$1               # name of the corpus (timit, mboshi, globalphone)
gsm_init=$2         # GSM for initialization
subset=$3           # subset of the corpus (mostly used for globalphone: FR, GE, ...)
outdir=$expdir/$db/$subset/hsubspace_aud_${feaname}_${ngauss}g_${prior}


# Create the directory structure.
mkdir -p $datadir $expdir $feadir


echo "--> Preparing data for the $db database"
local/$db/prepare_data.sh $datadir/$db $subset
for f in ali trans uttids wav.scp; do
    [ ! -d $datadir/$db/$subset/full/ ] && mkdir -p $datadir/$db/$subset/full/
    if [ ! -f $datadir/$db/$subset/full/$f ]; then
	cat $datadir/$db/$subset/*/$f | sort -u > $datadir/$db/$subset/full/$f
    fi
done


echo "--> Preparing pseudo-phones \"language\" information"
mkdir -p data/$db/$subset/lang_aud

# The option "--non-speech-unit" will force the decoder to start and end
# each utterance by a specific acoustic unit named "sil". The
# unit can also be freely decoded within the utterance. If your data
# is not well segmented and you are not sure that most of your data
# start and end with non-speech sound it's safer to remove this option.
if $nosil; then
    lang_opt=""
else
    lang_opt="--non-speech-unit"
fi
python utils/prepare_lang_aud.py \
    $lang_opt \
    $nunits > data/$db/$subset/lang_aud/units


for x in $train $test; do
    echo "--> Extracting features for the $db/$x database"
    steps/extract_features.sh \
	 --parallel-njobs $parallel_njobs \
        conf/${feaname}.yml \
        $datadir/$db/$subset/$x/wav.scp\ \
        $feadir/$db/$subset/$x

    # Create a "dataset". This "dataset" is just an object
    # associating the features with their utterance id and some
    # other meta-data (e.g. global mean, variance, ...).
    echo "--> Creating dataset(s) for $db database"
    steps/create_dataset.sh $datadir/$db/$subset/$x \
        $feadir/$db/$subset/$x/${feaname}.npz \
        $expdir/$db/$subset/datasets/$feaname/${x}.pkl
done


language=$db
[ ! -z $subset ] && language=${db}_${subset}

echo "--> Training the subspace Bayesian AUD system"
steps/hsubspace_aud.sh \
    $hsubspace_opts \
    --concentration $concentration \
    --parallel-opts "-l mem_free=1G,ram_free=1G -q all.q@@stable" \
    --parallel-njobs $parallel_njobs \
    --nosil $nosil \
    conf/hmm_${ngauss}g.yml \
    $gsm_init \
    data/$db/$subset/lang_aud \
    data/$db/$subset/$train \
    $expdir/$db/$subset/datasets/$feaname/${train}.pkl \
    $epochs $outdir $language


for x in $train $test; do
    score_outdir=$outdir/decode_perframe/$x
    echo "--> Decoding $db/$x dataset"
    steps/decode.sh \
        --per-frame \
        --parallel-opts "-l mem_free=2G,ram_free=2G -q all.q@@stable" \
        --parallel-njobs 30 \
        $outdir/final_${language}.mdl \
        data/$db/$subset/$x/uttids \
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



if $decode_all; then
    for ep in `seq 0 $decode_freq $epochs`; do
	[ ! -f $outdir/${ep}.mdl_$language ] && continue
	for x in $train $test; do
	    score_outdir=$outdir/decode_perframe_epoch$ep/$x
	    
	    echo "--> Decoding $db/$x dataset"
	    steps/decode.sh \
		--per-frame \
		--parallel-opts "-l mem_free=2G,ram_free=2G -q all.q@@stable" \
		--parallel-njobs 30 \
		$outdir/${ep}.mdl_$language \
		data/$db/$subset/$x/uttids \
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
    done
fi
