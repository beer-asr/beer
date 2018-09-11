#!/bin/bash

if [ $# -ne 1 ];then
    echo "$0: <setup_compare_llhs.sh>"
    exit 1
fi

setup=$1
. $setup

mkdir -p $outdir/tmp_ali
mkdir -p $outdir/tmp_hyp
tmpdir_ali=$(mktemp -d $outdir/tmp_ali/llhs.XXXX);
tmpdir_hyp=$(mktemp -d $outdir/tmp_hyp/llhs.XXXX);
trap "rm -rf $tmpdir_hyp" EXIT
trap "rm -rf $tmpdir_ali" EXIT
if [ $level == "frame" ]; then
    dir=${outdir}_frame
    cmd_ali="python utils/compute-likelihood.py \
            --align $alidir/ali_graphs.npz \
            $modeldir/final.mdl \
            $datadir/feats.npz \
            $tmpdir_ali"
    cmd_hyp="python utils/compute-likelihood.py \
            $modeldir/final.mdl \
            $datadir/feats.npz \
            $tmpdir_hyp"
elif [ $level == "phone" ]; then
    if [ ! -f $alidir/alis.npz ]; then
        steps/align_hmm.sh $setup $modeldir $datadir $alidir
    fi
    cmd_ali="python utils/compute-likelihood.py \
            --align $alidir/ali_graphs.npz \
            --smooth $alidir/alis.npz \
            --pdf2phone $modeldir/pdf_mapping.txt \
            $modeldir/final.mdl \
            $datadir/feats.npz $tmpdir_ali"
    cmd_hyp="python utils/compute-likelihood.py \
            --smooth $alidir/alis.npz \
            --pdf2phone $modeldir/pdf_mapping.txt \
            $modeldir/final.mdl \
            $datadir/feats.npz $tmpdir_hyp"
fi

for u in "ali" "hyp"; do
    cmd="cmd_${u}"
    tmpdir="tmpdir_${u}"
    mkdir -p $outdir/tmp_${u}
    echo "Computing log-likelihoods for $u model"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "comp_llhs" \
        "$parallel_opts" \
        "$njobs" \
        "$datadir/uttids" \
        "${!cmd}" \
        $outdir/$u || exit "Compute log-likelihood failed; see logs in $outdir/$u/log";
    zip -j $outdir/${u}_llhs.npz ${!tmpdir}/*.npy > /dev/null || exit 1
done

python utils/compute-det.py --thres $thres \
    $datadir/true_uttids \
    $datadir/wrong_uttids \
    $outdir/ali_llhs.npz \
    $outdir/hyp_llhs.npz \
    $outdir/fn_fp.txt
