#!/usr/bin/env bash

. path.sh
set -e

echo "$0 $@"

prior=gamma_dirichlet_process

sgd=false
nosil=true
gsm_init_epochs=1000
gsm_epochs=1000
concentration=50
cleanup=true
gsm_smoothing_epochs=5000
language_init_epochs=-10
gsm_std_lrate=1e-3
gsm_unit_latent_nsamples=10
gsm_lang_latent_nsamples=5
gsm_params_nsamples=5
opts_conf=""

parallel_env=sge
parallel_opts=""
parallel_njobs=20
nargs=8

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
      break
    esac
done

[ -f ./kutils/parse_options.sh ] && . ./kutils/parse_options.sh

if [ $# -ne $nargs ]; then
    echo "usage: $0 [OPTS] <hmm-conf> <hgsm-init-dir> <langdir> <datadir> <dataset> <epochs> <out-dir>"
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

optim_opts=""
[[ ! -z $opts_conf && -f $opts_conf ]] && . $opts_conf

modelconf=$1
gsm_init_dir=$2
langdir=$3
datadir=$4
dataset=$5
epochs=$6
outdir=$7
language=$8
mkdir -p $outdir

if $sgd; then
    optim_opts="$optim_opts --use-sgd"
fi

gsm_init=$gsm_init_dir/hgsm_final.mdl
[ ! -f $gsm_init ] && gsm_init=$gsm_init_dir/gsm_final.mdl
[ ! -f $gsm_init ] && \
    echo "gsm init file $gsm_init_dir/hgsm_final.mdl not found" && exit 1
units_init=$gsm_init_dir/units_posts_final.pkl


# Get the latent dimension from the GSM for initialization.
cmd="import pickle
with open(\"${gsm_init}\", \"rb\") as f:
    gsm = pickle.load(f)
print(gsm['lang_latent_dim'], gsm['unit_latent_dim'])
"
gsm_latent_dim=$(python -c "$cmd")
lang_latent_dim=$(echo $gsm_latent_dim | cut -d ' ' -f1)
unit_latent_dim=$(echo $gsm_latent_dim | cut -d ' ' -f2)
echo "language latent dimension: $lang_latent_dim"
echo "unit latent dimension: $unit_latent_dim"


# Create the units' HMM.
if [ ! -f $outdir/hmms.mdl ]; then
    beer hmm mkphones -d $dataset $modelconf $langdir/units \
        $outdir/hmms.mdl || exit 1
else
    echo "units' HMM already created. Skipping."
fi


# Create the phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    ploop_opt=""
    if ! $nosil; then
	ploop_opt="--start-end-group non-speech-unit"
    fi
    beer hmm mkphoneloopgraph $ploop_opt \
        $langdir/units $outdir/ploop_graph.pkl || exit 1
    beer hmm mkdecodegraph $outdir/ploop_graph.pkl $outdir/hmms.mdl \
        $outdir/decode_graph.pkl || exit 1
    beer hmm mkphoneloop --concentration $concentration \
	 --weights-prior $prior $outdir/decode_graph.pkl \
        $outdir/hmms.mdl $outdir/hmm_init.mdl || exit 1
else
    echo "Phone Loop model already created. Skipping."
fi


# Create the hierarchical subspace phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    echo "$outdir/hmm_init.mdl $language" \
	 > $outdir/init_phoneloop_to_lang.txt
    beer hshmm mksphoneloop \
         -g "speech-unit" \
	 -l $lang_latent_dim \
	 -t $unit_latent_dim \
         $modelconf $outdir/init_phoneloop_to_lang.txt \
         $outdir/tmp_hgsm.mdl $outdir/units_posts_0.pkl $outdir/init.mdl

    echo "using GSM ($gsm_init)"
    cp $gsm_init $outdir/hgsm_init.mdl || exit 1

    echo "initializing the phone-loop"
    beer -d hshmm init \
	 $language \
	 $outdir/tmp_hgsm.mdl $outdir/hgsm_init.mdl \
	 $outdir/units_posts_0.pkl \
	 $outdir/init.mdl \
         $outdir/0.mdl $outdir/hgsm_0.mdl || exit 1
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
    hgsm=hgsm_${epoch}.mdl
    posts=units_posts_${epoch}.pkl

    if [ $epoch -ge 1 ]; then
        echo "found existing model, starting training from epoch $((epoch + 1))"
    else
        echo "starting training..."
    fi

    while [ $((++epoch)) -le $epochs ]; do
        echo "epoch: $epoch"

        # Accumulate the statistics in parallel.
	if [ ! -f $outdir/epoch${epoch}/.done.acc ]; then
            cmd="beer hmm accumulate ${alis} $outdir/${mdl}_${language} \
                $dataset $outdir/epoch${epoch}/elbo_JOBID.pkl"
            utils/parallel/submit_parallel.sh \
		"$parallel_env" \
		"hmm-acc" \
		"$parallel_opts" \
		"$parallel_njobs" \
		"$datadir/uttids" \
		"$cmd" \
		$outdir/epoch${epoch}|| exit 1
	    touch $outdir/epoch${epoch}/.done.acc
	fi

        # Update the model' parameters.
        find $outdir/epoch${epoch} -name '*pkl' | \
            beer hmm update -o $outdir/optim_state.pth $outdir/${mdl}_${language} \
                $outdir/tmp.mdl 2>&1 | tee -a $outdir/training.log || exit 1

	echo "$outdir/tmp.mdl $language" > $outdir/${epoch}_sphoneloop_to_lang.txt
        if [ $epoch -eq $epochs ]; then
            train_epochs=$gsm_smoothing_epochs
	elif [ $epoch -eq 1 ]; then
	    train_epochs=$gsm_init_epochs
        else
            train_epochs=$gsm_epochs
        fi
	if [ $epoch -le $language_init_epochs ]; then
	    train_opts="$optim_opts --skip-unit-posterior"
	else
	    train_opts="$optim_opts"
	fi

        # Train the HGSM and the posteriors.
        cmd="beer -d hshmm train \
            --gpu \
 	    $train_opts \
            -o $outdir/hgsm_optim_state.pth \
	    -r 50 \
            --skip-root-subspace \
            --epochs $train_epochs \
            --learning-rate-std $gsm_std_lrate \
            --unit-latent-nsamples $gsm_unit_latent_nsamples \
            --lang-latent-nsamples $gsm_lang_latent_nsamples \
            --params-nsamples $gsm_params_nsamples \
            $outdir/$hgsm $outdir/$posts $outdir/${epoch}_sphoneloop_to_lang.txt \
            $outdir/hgsm_${epoch}.mdl $outdir/units_posts_${epoch}.pkl \
            $outdir/${epoch}.mdl"

        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "hgsm-training" \
            "-l gpu=1,gpu_ram=2G" \
            "$cmd" \
            $outdir/epoch${epoch} || exit 1


	if $cleanup; then
	    find $outdir/epoch${epoch} -name '*pkl' -delete
	fi
        mdl=${epoch}.mdl
        hgsm=hgsm_${epoch}.mdl
        posts=units_posts_${epoch}.pkl
    done

    cp $outdir/$mdl $outdir/final.mdl
    cp $outdir/$hgsm $outdir/hgsm_final.mdl
    cp $outdir/$posts $outdir/units_posts_final.pkl
    for lang in $language; do
	cp $outdir/${mdl}_${lang} $outdir/final_${lang}.mdl
    done
else
    echo "hierarchical subspace phone-loop already trained"
fi

