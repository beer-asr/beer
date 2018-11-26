#!/usr/bin/env bash

. path.sh

parallel_env=sge
parallel_opts=""
parallel_njobs=10

if [ $# -ne 6 ]; then
    echo "usage: <model-conf> <phone-mapping> <uttids> <dataset> <epochs> <out-dir>"
    exit 1
fi

modelconf=$1
phone_mapping=$2
uttids=$3
dataset=$4
epochs=$5
outdir=$6
mkdir -p $outdir


# Create the units' HMM.
steps/create_hmm.sh --outdir $outdir --mapping $phone_mapping \
    $modelconf $dataset $outdir/0.mdl || exit 1

exit 0


# Training.
if [ ! -f $outdir/final.mdl ]; then
    echo "training..."

    # Retrieve the last model.
    mdl=$(find $outdir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    epoch="${mdl%.*}"
    echo $epoch

    # ...and the optimizer.
    optim=optim_${epoch}.mdl

    while [ $((++epoch)) -le $epochs ]; do
        echo "epoch: $epoch"

        # Accumulate the statistics in parallel.
        cmd="beer hmm accumulate $outdir/$mdl $dataset \
             $outdir/epoch${epoch}/elbo_JOBID.pkl"
        utils/parallel/submit_parallel.sh \
            "$parallel_env" \
            "hmm-acc" \
            "$parallel_opts" \
            "$parallel_njobs" \
            "$uttids" \
            "$cmd" \
            $outdir/epoch${epoch}|| exit 1

        # Update the model' parameters.
        find $outdir/epoch${epoch} -name '*pkl' | \
            beer hmm update $outdir/$mdl $outdir/$optim \
                $outdir/${epoch}.mdl $outdir/optim_${epoch}.pkl || exit 1

        mdl=${epoch}.mdl
        optim=optim_${epoch}.pkl
    done

    cp $outdir/$mdl $outdir/final.mdl
else
    echo "Model already trained. Skipping."
fi

# Generating labels.
if [ ! -f $outdir/trans.txt ]; then
    # Creating the most likely transcription.
    echo "generating transcription for the $dataset dataset..."
    beer hmm decode --per-frame $outdir/final.mdl \
        $dataset > $outdir/trans.txt || exit 1
else
    echo "transcription already generated. Skipping."
fi

