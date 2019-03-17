########################################################################
# Set the virtual environment in which BEER is installed.
source activate beer


#######################################################################
# Setup CUDA.
export CUDA_VISIBLE_DEVICES=$(utils/free-gpus.sh)
#CUDA_LAUNCH_BLOCKING=1 # for debugging

