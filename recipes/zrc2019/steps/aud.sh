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
if [ ! -f $outdir/hmms.mdl ]; then
    beer hmm mkphones -d $dataset $modelconf $outdir/hmms.mdl || exit 1
else
    echo "units' HMM already created. Skipping."
fi


# Create the phone-loop model.
if [ ! -f $outdir/ploop_init.mdl ]; then
    beer hmm phonelist $outdir/hmms.mdl | \
        beer hmm mkphoneloopgraph -s sil - \
        $outdir/ploop_graph.pkl || exit 1
    beer hmm mkdecodegraph $outdir/ploop_graph.pkl $outdir/hmms.mdl \
        $outdir/decode_graph.pkl || exit 1
    beer hmm mkphoneloop $outdir/decode_graph.pkl $outdir/hmms.mdl \
        $outdir/ploop_init.mdl
else
    echo "Phone Loop model already created. Skipping."
fi


# Training.
if [ ! -f $outdir/final.mdl ]; then
    echo "training..."
    beer hmm train -l $lrate -b $bsize -e $epochs \
        $outdir/ploop_init.mdl $dataset $outdir/final.mdl
else
    echo "Model already trained. Skipping."
fi


# Creating the most likely transcription.
if [ ! -f $outdir/trans.txt ]; then
    echo "generating transcription for the $dataset dataset..."
    beer hmm decode $outdir/final.mdl $dataset > $outdir/trans.txt || exit 1
else
    echo "Transcription already generated. Skipping."
fi

