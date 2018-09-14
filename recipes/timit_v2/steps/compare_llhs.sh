#!/bin/bash

if [ $# -ne 1 ];then
    echo "$0: <setup_compare_llhs.sh>"
    exit 1
fi

setup=$1
. $setup
stage=-1


mkdir -p $outdir/tmp_ali
mkdir -p $outdir/tmp_hyp
tmpdir_ali=$(mktemp -d $outdir/tmp_ali/llhs.XXXX);
tmpdir_hyp=$(mktemp -d $outdir/tmp_hyp/llhs.XXXX);
trap "rm -rf $tmpdir_hyp" EXIT
trap "rm -rf $tmpdir_ali" EXIT
if [ $level == "frame" ]; then
    if [ ! -f $alidir/alis.npz ]; then
        steps/align_hmm.sh $setup $modeldir $datadir $alidir
    fi
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

if [ $stage -le 1 ]; then
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
        zip -j $outdir/${u}_${level}_llhs.npz ${!tmpdir}/*.npy > /dev/null || exit 1
    done
fi

if [ $stage -le 2 ]; then
    python utils/compute-llhs-diff.py \
        --diff-detail "var" \
        $outdir/ali_${level}_llhs.npz \
        $outdir/hyp_${level}_llhs.npz \
        $outdir/ali_hyp_llhs_diff_var.txt
fi

python utils/compute-det.py --thres $thres \
    $datadir/true_uttids \
    $datadir/false_uttids \
    $outdir/ali_hyp_llhs_diff_var.txt \
    $outdir/det_${level}_llhs.txt
