#!/usr/bin/env bash

set -e

. path.sh

gsm_latent_dim=10
gsm_init_epochs=1000
gsm_epochs=1000
gsm_smoothing_epochs=5000
gsm_std_lrate=1e-2
gsm_latent_nsamples=10
gsm_params_nsamples=5
gsm_classes=""

parallel_env=sge
parallel_opts=""
parallel_njobs=20
nargs=7

while [[ $# -gt $nargs ]]; do
    case $1 in
      --classes)
      gsm_classes="-c $2"
      shift
      shift
      ;;
      --latent-dim)
      gsm_latent_dim=$2
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
    echo "usage: $0 [OPTS] <hmm-conf> <gsm> <hmmdir> <datadir> <dataset> <epochs> <out-dir>"
    echo ""
    echo "Train a SHMM based AUD system."
    echo ""
    echo "Options:"
    echo "  Generalized Subspace Model:"
    echo "  --classes           units broad classes file (default: none)"
    echo "  --latent-dim        dimension of the subspace (default: 10)"
    echo ""
    echo "  Parallel environment:"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environmenti (default: "")"
    echo "  --parallel-njobs    number of parallel jobs to use (default: $parallel_njobs)"
    echo ""
    exit 1
fi

modelconf=$1
gsm=$2
hmmdir=$3
datadir=$4
dataset=$5
epochs=$6
outdir=$7
mkdir -p $outdir


# Create the subspace phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    cp $gsm $outdir/gsm_init.mdl || exit 1

    beer shmm mksphoneloop \
        $gsm_classes -g "speech-unit" -l $gsm_latent_dim \
        $modelconf $hmmdir/final.mdl \
        /dev/null $outdir/units_posts_init.pkl $outdir/init.mdl

    # Train the GSM and the posteriors.
    cmd="beer -d shmm train \
        --gpu \
        -o $outdir/gsm_optim_state.pth \
        --posteriors \
        --learning-rate-std $gsm_std_lrate \
        --epochs $gsm_init_epochs \
        --latent-nsamples $gsm_latent_nsamples \
        --params-nsamples $gsm_params_nsamples \
        $outdir/gsm_init.mdl $outdir/units_posts_init.pkl \
        $outdir/init.mdl $outdir/gsm_0.mdl $outdir/units_posts_0.pkl \
        $outdir/0.mdl"

    utils/parallel/submit_single.sh \
        "$parallel_env" \
        "gsm-pretraining" \
        "-l gpu=1,gpu_ram=2G" \
        "$cmd" \
        $outdir/pretraining || exit 1

else
    echo "subspace phone Loop model already created"
fi


# Check if the hmmdir has alignment graphs.
alis=""
if [ -f $hmmdir/alis.npz ]; then
    alis="--alis $hmmdir/alis.npz"
    echo "using alignments: $hmmdir/alis.npz"
fi

# Training.
if [ ! -f $outdir/final.mdl ]; then
    # Retrieve the last model.
    mdl=$(find $outdir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    echo "mdl: $mdl"
    epoch="${mdl%.*}"
    gsm=$outdir/gsm_${epoch}.mdl
    posts=$outdir/units_posts_${epoch}.pkl

    if [ $epoch -ge 1 ]; then
        echo "found existing model, starting training from epoch $((epoch + 1))"
    else
        echo "starting training..."
    fi

    while [ $((++epoch)) -le $epochs ]; do
        echo "epoch: $epoch"

        # Accumulate the statistics in parallel.
        cmd="beer hmm accumulate ${alis} $outdir/$mdl \
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
                $outdir/tmp.mdl 2>&1 | tee -a $outdir/training.log || exit 1

        if [ $epoch -eq $epochs ]; then
            train_epochs=$gsm_smoothing_epochs
        else
            train_epochs=$gsm_epochs
        fi

        # Train the GSM and the posteriors.
        cmd="beer -d shmm train \
            --gpu \
            -o $outdir/gsm_optim_state.pth \
            --posteriors \
            --epochs $train_epochs \
            --learning-rate-std $gsm_std_lrate \
            --latent-nsamples $gsm_latent_nsamples \
            --params-nsamples $gsm_params_nsamples \
            $gsm $posts $outdir/tmp.mdl \
            $outdir/gsm_${epoch}.mdl $outdir/units_posts_${epoch}.pkl \
            $outdir/${epoch}.mdl"

        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "gsm-training" \
            "-l gpu=1,gpu_ram=2G" \
            "$cmd" \
            $outdir/epoch${epoch} || exit 1


        mdl=${epoch}.mdl
    done

    cp $outdir/$mdl $outdir/final.mdl
else
    echo "subspace phone-loop already trained"
fi

