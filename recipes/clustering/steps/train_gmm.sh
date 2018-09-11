#!/bin/bash

# Train a GMM.


if [ $# -ne 3 ];then
    echo "$0: <setup.sh> <data-train-dir> <mdl-dir>"
    exit 1
fi

setup=$1
data_train_dir=$2
mdl_dir=$3
mkdir -p $mdl_dir


. $setup
[[ -f "$gmm_conf" ]] || \
    { echo "File not found: $hmm_conf"; exit 1; }



if [ ! -f $mdl_dir/0.mdl ]; then
    echo "Building the GMM"

    cat $gmm_conf | sed s/{size}/$gmm_size/g > $mdl_dir/gmm.yml

    # Create the HMM model.
    python utils/gmm-create.py \
        --stats $data_train_dir/stats.npz \
        $mdl_dir/gmm.yml \
        $mdl_dir/0.mdl || exit 1

else
    echo "Using previously created model: $mdl_dir/0.mdl"
fi


# Train the model.
if [ ! -f $mdl_dir/final.mdl ];then
    echo "Training"

    # Retrieve the last model.
    mdl=$(find $mdl_dir -name "[0-9]*mdl" -exec basename {} \; | \
        sort -t '.' -k 1 -g | tail -1)
    iter="${mdl%.*}"

    if [ $iter -ge 1 ]; then
        echo "Found existing model. Starting from iteration: $((iter + 1))"
    fi

    while [ $((++iter)) -le $gmm_train_iters ]; do
        echo "Iteration: $iter"

        cmd="python -u utils/gmm-train.py \
                --lrate $gmm_train_lrate \
                $gmm_train_opts \
                $mdl_dir/$((iter - 1)).mdl \
                $data_train_dir/batches \
                $data_train_dir/stats.npz \
                $mdl_dir/${iter}.mdl"
        utils/parallel/submit_single.sh \
            "$parallel_env" \
            "gmm-train-iter$iter" \
            "$gmm_train_parallel_opts" \
            "$cmd" \
            $mdl_dir || exit 1

        mdl=${iter}.mdl
    done

    ln -s $mdl_dir/$mdl $mdl_dir/final.mdl
else
    echo "Model already trained: $mdl_dir/final.mdl"
fi

