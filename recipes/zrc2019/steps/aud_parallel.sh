#!/usr/bin/env bash

. path.sh

parallel_env=sge
parallel_opts=""
parallel_njobs=10

if [ $# -ne 5 ]; then
    echo "usage: <model-conf> <uttids> <dataset> <epochs> <out-dir>"
    exit 1
fi

modelconf=$1
uttids=$2
dataset=$3
epochs=$4
outdir=$5
mkdir -p $outdir


# Create the units' HMM.
if [ ! -f $outdir/hmms.mdl ]; then
    beer hmm mkphones -d $dataset $modelconf $outdir/hmms.mdl || exit 1
else
    echo "units' HMM already created. Skipping."
fi


# Create the phone-loop model.
if [ ! -f $outdir/0.mdl ]; then
    beer hmm phonelist $outdir/hmms.mdl | \
        beer hmm mkphoneloopgraph - \
        $outdir/ploop_graph.pkl || exit 1
    beer hmm mkdecodegraph $outdir/ploop_graph.pkl $outdir/hmms.mdl \
        $outdir/decode_graph.pkl || exit 1
    beer hmm mkphoneloop $outdir/decode_graph.pkl $outdir/hmms.mdl \
        $outdir/0.mdl || exit 1

    # Create the optimizer of the training.
    beer hmm optimizer $outdir/0.mdl $outdir/optim_0.mdl || exit 1
else
    echo "Phone Loop model already created. Skipping."
fi


#beer hmm train -l $lrate -b $bsize -e $epochs \
#    $outdir/ploop_init.mdl $dataset $outdir/final.mdl

#cmd="python utils/hmm-align.py \
#--ali-graphs $mdl_dir/ali_graphs.npz \
#$mdl_dir/$mdl  $data_train_dir/feats.npz  $tmpdir"
#utils/parallel/submit_parallel.sh \
#"$parallel_env" \
#"hmm-align-iter$iter" \
#"$hmm_align_parallel_opts" \
#"$hmm_align_njobs" \
#"$data_train_dir/uttids" \
#"$cmd" \
#$mdl_dir || exit 1
#find $tmpdir -name '*npy' | \
#  zip -j -@ $mdl_dir/alis.npz > /dev/null || exit 1


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

    # Creating the most likely transcription.
    echo "generating transcription for the $dataset dataset..."
    beer hmm decode $outdir/final.mdl $dataset > $outdir/trans.txt || exit 1
else
    echo "Model already trained. Skipping."
fi




