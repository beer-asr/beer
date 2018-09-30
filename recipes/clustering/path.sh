#!/bin/sh

# Set the specific environment variable to run the recipe.
# By default, we assume that beer was installed on a specific
# python virtual enviroment named "beer".
source activate beer

export CUDA_VISIBLE_DEVICES=$(utils/free-gpus.sh)
