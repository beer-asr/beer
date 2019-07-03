#!/usr/bin/env bash

set -e

. path.sh

prior=gamma_dirichlet_process

gsm_init_epochs=1000
gsm_epochs=1000
gsm_smoothing_epochs=5000
gsm_std_lrate=1e-3
gsm_latent_nsamples=10
gsm_params_nsamples=5

parallel_env=sge
parallel_opts=""
parallel_njobs=20
nargs=7

while [[ $# -gt $nargs ]]; do
    case $1 in
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
    echo "usage: $0 [OPTS] <hmm-conf> <gsm-init-dir> <langdir> <datadir> <dataset> <epochs> <out-dir>"
    echo ""
    echo "Train a SHMM based AUD system."
    echo ""
    echo "Options:"
    echo "  Phone Loop Model:"
    echo "  --prior             type of prior [gamma_dirichlet_process|"
    echo "                      dirichlet_process|dirichlet] for the"
    echo "                      units weights (default:gamma_dirichlet_process)"
    echo ""
    echo "  Parallel environment:"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environmenti (default: "")"
    echo "  --parallel-njobs    number of parallel jobs to use (default: $parallel_njobs)"
    echo ""
    exit 1
fi

modelconf=$1
gsm_init_dir=$2
langdir=$3
datadir=$4
dataset=$5
epochs=$6
outdir=$7
mkdir -p $outdir

gsm_init=$gsm_init_dir/gsm_final.mdl
units_init=$gsm_init_dir/units_posts_final.pkl


# Get the latent dimension from the GSM for initialization.
cmd="import pickle
with open(\"${gsm_init}\", \"rb\") as f:
    gsm = pickle.load(f)
print(gsm.transform.in_dim)
"
gsm_latent_dim=$(python -c "$cmd")
echo "subspace latent dimension: $gsm_latent_dim"


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
    beer hmm mkphoneloop --weights-prior $prior $outdir/decode_graph.pkl \
        $outdir/hmms.mdl $outdir/hmm_init.mdl || exit 1
else
    echo "Phone Loop model already created. Skipping."
fi


# Create the subspace phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    beer shmm mksphoneloop \
        -g "speech-unit" -l $gsm_latent_dim \
        $modelconf $outdir/hmm_init.mdl \
        /dev/null $outdir/units_posts_0.pkl $outdir/init.mdl

    echo "using GSM ($gsm_init)"
    cp $gsm_init $outdir/gsm_0.mdl || exit 1

    echo "initializing the phone-loop"
    beer shmm init \
        $outdir/gsm_0.mdl  $outdir/units_posts_0.pkl $outdir/init.mdl \
        $outdir/0.mdl || exit 1
else
    echo "subspace phone Loop model already created"
fi


# Training.
if [ ! -f $outdir/final.mdl ]; then
    # Retrieve the last model.
    mdl=$(find $outdir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    echo "mdl: $mdl"
    epoch="${mdl%.*}"
    gsm=gsm_${epoch}.mdl
    posts=units_posts_${epoch}.pkl

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
            $outdir/$gsm $outdir/$posts $outdir/tmp.mdl \
            $outdir/gsm_${epoch}.mdl $outdir/units_posts_${epoch}.pkl \
            $outdir/${epoch}.mdl"

        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "gsm-training" \
            "-l gpu=1,gpu_ram=2G" \
            "$cmd" \
            $outdir/epoch${epoch} || exit 1


        mdl=${epoch}.mdl
        gsm=gsm_${epoch}.mdl
        posts=units_posts_${epoch}.pkl
    done

    cp $outdir/$mdl $outdir/final.mdl
    cp $outdir/$gsm $outdir/gsm_final.mdl
    cp $outdir/$posts $outdir/units_posts_final.pkl
else
    echo "subspace phone-loop already trained"
fi

