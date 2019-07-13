#!/usr/bin/env bash

parallel_env=sge
parallel_opts=""
parallel_njobs=4
nargs=3

while [[ $# -gt $nargs ]]; do
    case $1 in
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
    echo "usage: $0 [OPTS] <fea-conf> <data-dir> <out-fea-dir>"
    echo ""
    echo "Extract features"
    echo ""
    echo "Options:"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environment"
    echo "  --parallel-njobs    number of parallel jobs to use"
    echo ""
    exit 1
fi

if [ $# -ne 3 ]; then
    echo "usage: <fea-conf> <data-dir> <out-fea-dir>"
    exit 1
fi

conf=$1
datadir=$2
feadir=$3
feaname=$(basename $conf)
feaname="${feaname%.*}"

mkdir -p $feadir


if [ ! -f $feadir/${feaname}.npz ]; then
    echo "extracting features in $feadir/${feaname}.npz"

    mkdir -p $feadir/${feaname}_tmp

    # Extract the features.
    cmd="beer features extract $conf - $feadir/${feaname}_tmp"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "extract-features" \
        "$parallel_opts" \
        "$parallel_njobs" \
        "$datadir/wav.scp" \
        "$cmd" \
        $feadir || exit 1

    # Put all the features files into a single archive.
    beer features archive $feadir/${feaname}_tmp $feadir/${feaname}.npz

    # We don't need the original features anymore as they are stored in
    # the archive.
    rm -fr $feadir/${feaname}_tmp
else
    echo "Features already extracted ($feadir/${feaname}.npz). Skipping."
fi

