#!/bin/bash


cmd="/export/b07/jyang/kaldi-jyang/kaldi/src/featbin/copy-feats"
scp='/export/b07/jyang/beer/recipes/timit/data/test/feats_kaldi.scp'
outdir='/export/b07/jyang/beer/recipes/timit/data/test/feats_kaldi'

if [ ! -d $outdir ];then
    mkdir -p $outdir
fi

while read l
    do
    echo "$l" | $cmd scp:- ark,t:- | \
        python3 convert_kaldi_MFCC_to_npz.py "$outdir/" 
done < $scp
