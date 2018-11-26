#!/usr/bin/env bash

if [ $# -ne 6 ]; then
    echo "usage: <model-conf> <dataset> <epochs> <learning-rate> <batch-size> <out-dir>"
    exit 1
fi

modelconf=$1
dataset=$2
epochs=$3
lrate=$4
bsize=$5
outdir=$6
mkdir -p $outdir


# Create the units' HMM.
steps/create_hmm.sh --outdir $outdir $modelconf $dataset \
    $outdir/ploop_init.mdl || exit 1

# Training.
if [ ! -f $outdir/final.mdl ]; then
    echo "training..."
    beer hmm train -l $lrate -b $bsize -e $epochs \
        $outdir/ploop_init.mdl $dataset $outdir/final.mdl
else
    echo "Model already trained. Skipping."
fi

# Generating labels.
if [ ! -f $outdir/trans.txt ]; then
    # Creating the most likely transcription.
    echo "generating transcription for the $dataset dataset..."
    beer hmm decode --per-frame $outdir/final.mdl \
        $dataset > $outdir/trans.txt || exit 1
else
    echo "transcription already generated. Skipping."
fi

