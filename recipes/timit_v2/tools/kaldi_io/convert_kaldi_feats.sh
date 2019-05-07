#!/bin/bash

if [ $# -ne 2 ];then
    echo "$0: <feat-dir> <output-dir>"
    exit 1
fi
. tools/kaldi_parallel/cmd.sh
featdir=$1
outdir=$2
nj=10

if [ ! -f $outdir/feats.npz ];then
    mkdir -p $outdir/tmp
fi

split --numeric-suffixes=1 -e -n l/$nj $featdir/feats.scp $outdir/tmp/

pushd $outdir/tmp
for f in `ls 0*`; do
    mv $f `sed "s/0//" <<< $f`
done

popd
#cmd=$decode_cmd
cmd="tools/kaldi_parallel/run.pl"

$cmd JOB=1:$nj $outdir/log/convert_fts.JOB.log \
    python3 tools/kaldi_io/read_kaldi_feats_scp.py $outdir/tmp/JOB $outdir || exit 1

zip -j $featdir/feats.npz $outdir/*.npy > /dev/null || exit 1
#rm -r $outdir
