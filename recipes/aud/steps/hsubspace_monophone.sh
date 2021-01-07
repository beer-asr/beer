#!/usr/bin/env bash

set -e

echo "$0 $@"
. path.sh

cleanup=true
unit_latent_dim=
sgd=false
gsm_unit_latent_dim=10
lang_latent_dim=
gsm_lang_latent_dim=2
gsm_init_epochs=15000
gsm_epochs=1000
gsm_smoothing_epochs=5000
gsm_std_lrate=1e-3
gsm_unit_latent_nsamples=10
gsm_lang_latent_nsamples=5
gsm_params_nsamples=5
gsm_classes=""
opts_conf=""

parallel_env=sge
parallel_opts=""
parallel_njobs=20
nargs=6


while [[ $# -gt $nargs ]]; do
    case $1 in
      --classes)
      gsm_classes="-c $2"
      shift
      shift
      ;;
      --unit-latent-dim)
      gsm_unit_latent_dim=$2
      shift
      shift
      ;;
      --lang-latent-dim)
      gsm_lang_latent_dim=$2
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
      ;;
    esac
done

[ -f ./kutils/parse_options.sh ] && . ./kutils/parse_options.sh

[ ! -z $unit_latent_dim ] && gsm_unit_latent_dim=$unit_latent_dim
[ ! -z $lang_latent_dim ] && gsm_lang_latent_dim=$lang_latent_dim
if [ $# -ne $nargs ]; then
    echo "usage: $0 [OPTS] <hmm-conf> <hmmdir> <datadir> <dataset> <epochs> <out-dir>"
    echo ""
    echo "Train a HMM based monophone phone recognizer system."
    echo ""
    echo "Options:"
    echo "  Generalized Subspace Model:"
    echo "  --classes           units broad classes file (default: none)"
    echo "  --unit-latent-dim   dimension of the unit subspace (default: 10)"
    echo "  --lang-latent-dim   dimension of the language subspace (default: 2)"
    echo ""
    echo "  Parallel environment:"
    echo "  --parallel-env      parallel environment to use (default:sge)"
    echo "  --parallel-opts     options to pass to the parallel environmenti (default: "")"
    echo "  --parallel-njobs    number of parallel jobs to use (default: $parallel_njobs)"
    echo ""
    exit 1
fi

[[ ! -z $opts_conf && -f $opts_conf ]] && . $opts_conf

modelconf=$1
hmmdir=$2 # Space separated list of language-specifi hmm directories
datadir=$3 # Space separated list of language-specific datadirs: lang1,dir1 lang2,dir2 ... langN,dirN
dataset=$4 # Similarly space separated list of dataset objects
epochs=$5
outdir=$6
mkdir -p $outdir

optim_opts=""
if $sgd; then
    optim_opts="--use-sgd"
fi
[ -f $outdir/optim_opts ] && optim_opts=`cat $outdir/optim_opts`

langs=""
declare -A datasets_arr
declare -A datadirs_arr
declare -A hmmdirs_arr

for x in $dataset; do
    lang=${x%,*}
    path=${x#*,}
    datasets_arr[$lang]=$path
    langs="$langs $lang"
done

for x in $datadir; do
    lang=${x%,*}
    path=${x#*,}
    datadirs_arr[$lang]=$path
done

for x in $hmmdir; do
    lang=${x%,*}
    path=${x#*,}
    hmmdirs_arr[$lang]=$path
done

for lang in $langs; do
    [ -z ${datadirs_arr[$lang]} ] && echo "datadir no set for $lang" && exit 1
    [ -z ${hmmdirs_arr[$lang]} ] && echo "hmmdir no set for $lang" && exit 1
done

# Create the hierarchical subspace phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    for lang in ${!hmmdirs_arr[@]}; do
	echo "${hmmdirs_arr[$lang]}/final.mdl $lang"
    done > $outdir/init_phoneloop_to_lang.txt

    beer hshmm mksphoneloop \
         -g "speech-unit" \
	 -l $gsm_lang_latent_dim -t $gsm_unit_latent_dim \
         $modelconf \
	 $outdir/init_phoneloop_to_lang.txt\
         $outdir/hgsm_init.mdl $outdir/units_posts_init.pkl $outdir/init.mdl

    for lang in ${!hmmdirs_arr[@]}; do
	echo "$outdir/init.mdl_$lang $lang"
    done > $outdir/init_sphoneloop_to_lang.txt

    # Initialize the HGSM and posteriors as a "UBM".
    cmd="beer -d hshmm train \
        --gpu \
        $optim_opts \
        -o $outdir/hgsm_optim_state.pth \
        --learning-rate-std $gsm_std_lrate \
        --epochs $gsm_init_epochs \
        --unit-latent-nsamples $gsm_unit_latent_nsamples \
	--lang-latent-nsamples $gsm_lang_latent_nsamples \
        --params-nsamples $gsm_params_nsamples \
        $outdir/hgsm_init.mdl $outdir/units_posts_init.pkl \
        $outdir/init_sphoneloop_to_lang.txt $outdir/hgsm_0.mdl $outdir/units_posts_0.pkl \
        $outdir/0.mdl"
    echo $optim_opts > $outdir/optim_opts

    utils/parallel/submit_single.sh \
        "$parallel_env" \
        "hgsm-pretraining" \
        "-l mem_free=20G,ram_free=20G,gpu=1,gpu_ram=8G -q long.q" \
        "$cmd" \
        $outdir/pretraining || exit 1

else
    echo "hierarchical subspace phone loop model already created"
fi


# Check if the hmmdir has alignment graphs.
alis=""
for hmmdir_ in ${hmmdirs_arr[@]}; do
    if [ -f ${hmmdir_}/alis.npz ]; then
	alis="--alis $hmmdir_/alis.npz"
	echo "will use alignments: $hmmdir_/alis.npz"
    fi
done

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
	[ -f $outdir/${epoch}_sphoneloop_to_lang.txt ] && rm $outdir/${epoch}_sphoneloop_to_lang.txt
	for lang in $langs; do
	    hmmdir_=${hmmdirs_arr[$lang]}
	    if [ -f ${hmmdir_}/alis.npz ]; then
		alis="--alis $hmmdir_/alis.npz"
	    else
		alis=""
	    fi
	    if [ ! -f $outdir/epoch${epoch}/$lang/.done.acc ]; then
		cmd="beer hmm accumulate ${alis} $outdir/${mdl}_${lang} \
                    	  ${datasets_arr[$lang]} $outdir/epoch${epoch}/$lang/elbo_JOBID.pkl"
		utils/parallel/submit_parallel.sh \
		    "$parallel_env" \
		    "hmm-acc" \
		    "$parallel_opts" \
		    "$parallel_njobs" \
		    "${datadirs_arr[$lang]}/uttids" \
		    "$cmd" \
		    $outdir/epoch${epoch}/$lang && touch $outdir/epoch${epoch}/$lang/.done.acc || exit 1 &
	    fi
	done
	wait
	
	for lang in $langs; do
            # Update the model' parameters.
            find $outdir/epoch${epoch}/$lang -name '*pkl' | \
		beer hmm update -o $outdir/optim_state.pth $outdir/${mdl}_${lang} \
                     $outdir/${lang}_tmp.mdl 2>&1 | tee -a $outdir/training_${lang}.log && \
		echo "$outdir/${lang}_tmp.mdl $lang" >> $outdir/${epoch}_sphoneloop_to_lang.txt || exit 1 &
	done
	wait

        if [ $epoch -eq $epochs ]; then
            train_epochs=$gsm_smoothing_epochs
        else
            train_epochs=$gsm_epochs
        fi

        # Train the HGSM and the posteriors.
        cmd="beer -d hshmm train \
            --gpu \
            $optim_opts \
            -o $outdir/hgsm_optim_state.pth \
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


        mdl=${epoch}.mdl
        hgsm=hgsm_${epoch}.mdl
        posts=units_posts_${epoch}.pkl
	if $cleanup; then
            for lang in $langs; do
	        find $outdir/epoch${epoch}/$lang -name '*pkl' -delete
            done
	fi
    done

    cp $outdir/$mdl $outdir/final.mdl
    cp $outdir/$hgsm $outdir/hgsm_final.mdl
    cp $outdir/$posts $outdir/units_posts_final.pkl
    for lang in $langs; do
	cp $outdir/${mdl}_${lang} $outdir/final_${lang}.mdl
    done
else
    echo "hierarchical subspace phone-loop already trained"
fi
