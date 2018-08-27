#!/bin/bash

if [ $# -ne 4 ];then
    echo "$0: <setup.sh> <model-dir> <data-dir> <align-dir>"
    exit 1
fi
setup=$1
. $setup
mdldir=$2
datadir=$3
alidir=$4


[ ! -d $alidir ] && mkdir -p $alidir/tmp

if [ ! -f $alidir/alis.npz ]; then
    if [ ! -f $alidir/ali_graphs.npz ]; then
        echo "Prepareing the alignment graph ..."
        tmpgraphdir=$(mktemp -d $alidir/tmp/graph.XXXX);
        trap "rm -rf $tmpgraphdir" EXIT
        cmd="python utils/prepare-alignments.py \
            $mdldir/phones_hmm.graphs $tmpgraphdir"
         utils/parallel/submit_parallel.sh \
             "$parallel_env" \
             "prepare-align" \
             "$hmm_align_parallel_opts" \
             "$hmm_align_njobs" \
             "$datadir/trans" \
             "$cmd" \
             $alidir || exit 1
         zip -j $alidir/ali_graphs.npz $tmpgraphdir/*.npy > /dev/null || exit 1
    fi
    echo "Begin to align ... "
    tmpdir=$(mktemp -d $alidir/tmp/ali.XXXX);
    trap "rm -rf $tmpdir" EXIT
    cmd="python utils/hmm-align.py \
        --ali-graphs $alidir/ali_graphs.npz \
        $mdldir/final.mdl  $datadir/feats.npz  $tmpdir"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "align" \
        "$hmm_align_parallel_opts" \
        "$hmm_align_njobs" \
        "$datadir/uttids" \
        "$cmd" \
        $alidir || exit 1
    zip -j $alidir/alis.npz $tmpdir/*.npy > /dev/null || exit 1
else
    echo "Alignments already in $alidir/alis.npz"
    exit 1
fi
