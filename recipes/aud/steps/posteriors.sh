#!/usr/bin/env bash

. path.sh

logdomain=''
acoustic_scale=1.
parallel_env=sge
parallel_opts=""
parallel_njobs=4
nargs=4

while [[ $# -gt $nargs ]]; do
    case $1 in
      --acoustic-scale)
      acoustic_scale=$2
      shift
      shift
      ;;
      --log-domain)
      logdomain='--log'
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
    echo "usage: $0 [OPTS] <model> <datadir> <dataset> <out-dir>"
    echo ""
    echo "Compute the phone posteriors."
    echo ""
    echo "Options:"
    echo "  --acoustic-scale    acoustic model scaling factor (default: 1)"
    echo "  --log-domain        store the posteriors in the log domain (default: false)"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environment"
    echo "  --parallel-njobs    number of parallel jobs to use"
    echo ""
    exit 1
fi

model=$1
datadir=$2
dataset=$3
outdir=$4
mkdir -p $outdir

if [ ! -f $outdir/posts.npz ]; then
    tmpdir=$(mktemp -d $outdir/tmp.XXX);
    trap 'rm -rf "$tmpdir"' EXIT
    cmd="beer hmm posteriors $logdomain -s $acoustic_scale --utts - \
         $model $dataset $tmpdir"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "hmm-posteriors" \
        "$parallel_opts" \
        "$parallel_njobs" \
        "$datadir/uttids" \
        "$cmd" \
        $outdir/compute_posts || exit 1
    find $tmpdir -name '*npy' | \
        zip -@ -j --quiet $outdir/posts.npz || exit 1
else
    echo "posteriors already computed"
fi

