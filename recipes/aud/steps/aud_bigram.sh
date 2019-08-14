#!/usr/bin/env bash

. path.sh

acoustic_scale=1.
prior=gamma_dirichlet_process
parallel_env=sge
parallel_opts=""
parallel_njobs=20
nargs=5

while [[ $# -gt $nargs ]]; do
    case $1 in
      --acoustic-scale)
      acoustic_scale=$2
      shift
      shift
      ;;
      --prior)
      prior=$2
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
    echo "usage: $0 [OPTS] <ploop-init> <datadir> <dataset> <epochs> <out-dir>"
    echo ""
    echo "Train a 2-gram HMM based Acoustic Unit Discovery (AUD) system."
    echo ""
    echo "Options:"
    echo "  --acoustic-scale    acoustic model scaling factor (default: 1)"
    echo "  --prior             type of prior [gamma_dirichlet_process|"
    echo "                      dirichlet_process|dirichlet] for the"
    echo "                      units weights (default:gamma_dirichlet_process)"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environment"
    echo "  --parallel-njobs    number of parallel jobs to use"
    echo ""
    exit 1
fi

ploopinit=$1
datadir=$2
dataset=$3
epochs=$4
outdir=$5
mkdir -p $outdir


# Create the 2-gram phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    beer hmm mkphoneloopbigram --weights-prior $prior $ploopinit \
        $outdir/0.mdl || exit 1
else
    echo "2-gram phone-loop model already created"
fi

# Training.
if [ ! -f $outdir/final.mdl ] || [ ! -f $outdir/${epochs}.mdl ]; then
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
        cmd="beer hmm accumulate -s $acoustic_scale \
             $outdir/$mdl $dataset \
             $outdir/epoch${epoch}/elbo_JOBID.pkl"
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

