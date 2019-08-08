#!/usr/bin/env bash

parallel_env=sge
parallel_opts=""
parallel_njobs=4
nargs=2

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
    echo "usage: $0 [OPTS] <scp> <out-fea-dir>"
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


scp=$1
feadir=$2
feaname=mbn

mkdir -p $feadir


if [ ! -f $feadir/${feaname}.npz ]; then
    echo "extracting features in $feadir/${feaname}.npz"

    mkdir -p $feadir/${feaname}_tmp

    # Extract the features.
    cmd="audio2bottleneck --outdir $feadir/${feaname}_tmp -"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "extract-mbn-features" \
        "$parallel_opts" \
        "$parallel_njobs" \
        "$scp" \
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

