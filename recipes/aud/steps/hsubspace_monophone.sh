#!/usr/bin/env bash

set -e

. path.sh

gsm_unit_latent_dim=10
gsm_lang_latent_dim=2
gsm_init_epochs=15000
gsm_epochs=1000
gsm_smoothing_epochs=5000
gsm_std_lrate=1e-3
gsm_unit_latent_nsamples=10
gsm_lang_latent_nsamples=5
gsm_params_nsamples=5
gsm_classes=""

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
      echo "unknown option: $1"
      exit 1
    esac
done

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

modelconf=$1
hmmdir=$2 # Space separated list too
datadir=$3 # Space separated list of language-specific datadirs: lang1,dir1 lang2,dir2 ... langN,dirN
dataset=$4 # Similarly space separated list of dataset objects
epochs=$5
outdir=$6
mkdir -p $outdir


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
# Create the subspace phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    for lang in ${!hmmdirs_arr[@]}; do
	echo "${hmmdirs_arr[$lang]}/final.mdl $lang"
    done > $outdir/init_phoneloop_to_lang.txt
    #$hmmdir/$lang/final.mdl
    beer hshmm mksphoneloop \
         -g "speech-unit" \
	 -l $gsm_lang_latent_dim -t $gsm_unit_latent_dim \
         $modelconf \
	 $outdir/init_phoneloop_to_lang.txt\
         $outdir/gsm_init.mdl $outdir/units_posts_init.pkl $outdir/init.mdl
    for lang in ${!hmmdirs_arr[@]}; do
	echo "$outdir/init.mdl_$lang $lang"
    done > $outdir/init_sphoneloop_to_lang.txt
    # Train the GSM and the posteriors.
    cmd="beer -d hshmm train \
        --gpu \
        -o $outdir/gsm_optim_state.pth \
        --learning-rate-std $gsm_std_lrate \
        --epochs $gsm_init_epochs \
        --unit-latent-nsamples $gsm_unit_latent_nsamples \
	--lang-latent-nsamples $gsm_lang_latent_nsamples \
        --params-nsamples $gsm_params_nsamples \
        $outdir/gsm_init.mdl $outdir/units_posts_init.pkl \
        $outdir/init_sphoneloop_to_lang.txt $outdir/gsm_0.mdl $outdir/units_posts_0.pkl \
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
		    $outdir/epoch${epoch}/$lang || exit 1
		touch $outdir/epoch${epoch}/$lang/.done.acc
	    fi
	

            # Update the model' parameters.
            find $outdir/epoch${epoch}/$lang -name '*pkl' | \
		beer hmm update -o $outdir/optim_state.pth $outdir/${mdl}_${lang} \
                     $outdir/${lang}_tmp.mdl 2>&1 | tee -a $outdir/training_${lang}.log || exit 1
	    echo "$outdir/${lang}_tmp.mdl $lang" >> $outdir/${epoch}_sphoneloop_to_lang.txt

	done
	#> $outdir/${epoch}_sphoneloop_to_lang.txt
        if [ $epoch -eq $epochs ]; then
            train_epochs=$gsm_smoothing_epochs
        else
            train_epochs=$gsm_epochs
        fi

        # Train the GSM and the posteriors.
        cmd="beer -d hshmm train \
            --gpu \
            -o $outdir/gsm_optim_state.pth \
            --epochs $train_epochs \
            --learning-rate-std $gsm_std_lrate \
            --unit-latent-nsamples $gsm_unit_latent_nsamples \
	    --lang-latent-nsamples $gsm_lang_latent_nsamples \
            --params-nsamples $gsm_params_nsamples \
            $outdir/$gsm $outdir/$posts $outdir/${epoch}_sphoneloop_to_lang.txt \
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
    for lang in $langs; do
	cp $outdir/${mdl}_${lang} $outdir/final_${lang}.mdl
    done
else
    echo "subspace phone-loop already trained"
fi

