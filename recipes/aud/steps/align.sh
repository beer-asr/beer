#!/usr/bin/env bash

. path.sh

acoustic_scale=1.
parallel_env=sge
parallel_opts=""
parallel_njobs=4
nargs=5

while [[ $# -gt $nargs ]]; do
    case $1 in
      --acoustic-scale)
      acoustic_scale=$2
      shift
      shift
      ;;
      --parallel-env)
      parallel_env=$2
      shift
      shift
      ;;
      --parallel-opts)
      parallel_opts=$2
      shift
      shift
      ;;
      --parallel-njobs)
      parallel_njobs=$2
      shift
      shift
      ;;
      *)
      echo "unknown option: $1"
      exit 1
    esac
done

if [ $# -ne $nargs ]; then
    echo "usage: $0 [OPTS] <model> <alis> <datadir> <dataset> <out-dir>"
    echo ""
    echo "Phone decoder"
    echo ""
    echo "Options:"
    echo "  --acoustic-scale    acoustic model scaling factor (default: 1)"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environment"
    echo "  --parallel-njobs    number of parallel jobs to use"
    echo ""
    exit 1
fi

model=$1
alis=$2
datadir=$3
dataset=$4
outdir=$5
mkdir -p $outdir

if [ ! -f $outdir/trans ]; then
    echo "decoding $dataset dataset..."

    cmd="beer hmm decode --per-frame -a $alis -s $acoustic_scale --utts - \
         $model $dataset >$outdir/trans_JOBID"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "hmm-decode" \
        "$parallel_opts" \
        "$parallel_njobs" \
        "$datadir/uttids" \
        "$cmd" \
        $outdir || exit 1

    cat $outdir/trans_* | sort > $outdir/trans || exit 1
else
    echo "data already aligned"
fi

