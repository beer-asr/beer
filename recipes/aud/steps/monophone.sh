#!/usr/bin/env bash

. path.sh

parallel_env=sge
parallel_opts=""
parallel_njobs=20
nargs=6

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
    echo "usage: $0 [OPTS] <hmm-conf> <langdir> <datadir> <dataset> <epochs> <out-dir>"
    echo ""
    echo "Train a HMM based monophone phone recognizer system."
    echo ""
    echo "Options:"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environmenti (default: "")"
    echo "  --parallel-njobs    number of parallel jobs to use (default: 4)"
    echo ""
    exit 1
fi

modelconf=$1
langdir=$2
datadir=$3
dataset=$4
epochs=$5
outdir=$6
mkdir -p $outdir


# Create the units' HMM.
if [ ! -f $outdir/hmms.mdl ]; then
    beer hmm mkphones -d $dataset $modelconf $langdir/units \
        $outdir/hmms.mdl || exit 1
else
    echo "units' HMM already created. Skipping."
fi


# Create the phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    beer hmm mkphoneloopgraph --start-end-group "non-speech-unit" \
        $langdir/units $outdir/ploop_graph.pkl || exit 1
    beer hmm mkdecodegraph $outdir/ploop_graph.pkl $outdir/hmms.mdl \
        $outdir/decode_graph.pkl || exit 1
    beer hmm mkphoneloop $outdir/decode_graph.pkl $outdir/hmms.mdl \
        $outdir/0.mdl || exit 1
else
    echo "Phone Loop model already created. Skipping."
fi


# Create the alignment graphs.
if [ ! -f $outdir/alis.npz ]; then
    mkdir -p $outdir/aligraphs

    # Accumulate the statistics in parallel.
    cmd="beer hmm mkaligraph $outdir/hmms.mdl $outdir/aligraphs"
    utils/parallel/submit_parallel.sh \
        "$parallel_env" \
        "compile-ali-graph" \
        "$parallel_opts" \
        "$parallel_njobs" \
        "$datadir/trans" \
        "$cmd" \
        $outdir/aligraphs || exit 1

    find $outdir/aligraphs -name '*npy' | \
        zip -@ -j --quiet $outdir/alis.npz || exit 1

    # We don't remove the directory to keep the log files.
    rm -fr $outdir/aligraphs/*npy
else
    echo "Alginments graphs already created. Skipping."
fi


# Training.
if [ ! -f $outdir/final.mdl ]; then
    # Retrieve the last model.
    mdl=$(find $outdir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    epoch="${mdl%.*}"

    if [ $epoch -ge 1 ]; then
        echo "found existing model, starting training from epoch $((epoch + 1))"
    else
        echo "starting training..."
    fi

    while [ $((++epoch)) -le $epochs ]; do
        echo "epoch: $epoch"

        # Accumulate the statistics in parallel.
        cmd="beer hmm accumulate --alis $outdir/alis.npz $outdir/$mdl \
                $dataset $outdir/epoch${epoch}/elbo_JOBID.pkl"
        utils/parallel/submit_parallel.sh \
            "$parallel_env" \
            "hmm-acc" \
            "$parallel_opts" \
            "$parallel_njobs" \
            "$datadir/uttids" \
            "$cmd" \
            $outdir/epoch${epoch}|| exit 1

        # Update the model' parameters.
        find $outdir/epoch${epoch} -name '*pkl' | \
            beer hmm update -o $outdir/optim_state.pth $outdir/$mdl \
                $outdir/${epoch}.mdl 2>&1 | \
                tee -a $outdir/training.log || exit 1

        mdl=${epoch}.mdl
    done

    cp $outdir/$mdl $outdir/final.mdl
else
    echo "Model already trained. Skipping."
fi

