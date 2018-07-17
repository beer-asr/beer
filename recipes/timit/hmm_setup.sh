#!/bin/bash

# Required
datadir="/export/b07/jyang/beer/recipes/timit/data/train/"
model_dir="/export/b07/jyang/beer/recipes/timit/exp_feat_transed"
phonelist="/export/b07/jyang/beer/recipes/timit/data/lang/phones.txt"
nstate_per_phone=3

feats=$datadir/feats_transed.npz
labels=$datadir/states.int.npz
feat_stats=$datadir/feats_stats.npz
emissions=$model_dir/emission.mdl
hmm_model_dir=$model_dir
nphones=`wc -l $phonelist | cut -d " " -f1`
nstates=$(($nphones * $nstate_per_phone))

# Optional
emission_type='norm_diag'
#training_type='baum_welch'
training_type='viterbi'
noise_std=0
lrate=0.1
batch_size=400
epochs=10
use_gpu=""
#use_gpu="--use-gpu"
