#!/bin/bash

# Required
modeldir="/export/b07/jyang/beer/recipes/timit/exp_feat_transformed_hmm_gmm_1/"
decode_data_dir="/export/b07/jyang/beer/recipes/timit/data/test"
feats=$decode_data_dir/feats_transformed.npz
trans=$decode_data_dir/phones.text
decode_dir=$modeldir/decode_test/
model=$modeldir/hmm.mdl
phonelist="/export/b07/jyang/beer/recipes/timit/data/lang/phones.txt"
nstate_per_phone=3

# Optional
gamma=0.5
#use_gpu=""
phone_39="/export/b07/jyang/beer/recipes/timit/data/lang/phones_48_to_39.map"
remove_sys="sil"
score="True"
