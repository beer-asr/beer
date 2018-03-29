#!/usr/bin/env bash

# beer installation script. This script assumes you have installed
# conda (see https://conda.io/docs).

# Name of the python virtual environment to create.
envname=beer

# Python version of the virtual environment.
python_version=3.6

# Operating system (only Linux an Darwin are supported).
os=$(uname -s)


# Check for the conda command.
conda=$(which conda) || \
if [ -z ${conda} ]; then
    echo "error: No 'conda' command."
    echo "Make sure anaconda is installed and the PATH variable is properly set."
    exit 1
fi
echo "Found: ${conda}"


# Create the new virtual environment environement.
if [ -n "$(${conda} info --envs | grep ${envname})" ]; then
    echo "The virtual environment \"${envname}\" already exists."
else
    export CMAKE_PREFIX_PATH="$(dirname ${conda})/../"
    conda create -y --name ${envname} python=${python_version} numpy \
        scipy numpy bokeh pyyaml mkl mkl-include setuptools cmake \
        cffi typing
fi


# Activate the new environment.
source activate ${envname}



# Install dependencies.
export CMAKE_PREFIX_PATH="$(dirname ${conda})/../"
conda install -y numpy scipy numpy bokeh pyyaml mkl mkl-include \
    setuptools cmake cffi typing


# If on linux, add support for GPU.
if [ ${os} = "Linux" ]; then
    conda install -y -c pytorch magma-cuda80
else
    echo "pytorch has no GPU support for OSX."
fi


# Download pytorch sources.
if [ ! -d "./pytorch" ]; then
    git clone --recursive https://github.com/pytorch/pytorch
else
    echo "pytorch sources already downloaded. Skipping."
fi


# Install pytorch.
cd pytorch
if [ -n "$(python -c "import torch" 2>&1 | grep Error)" ]; then
    if [ ${os} = 'Linux' ]; then
        python setup.py install
    elif [ ${os} = 'Darwin' ]; then
        MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ \
            python setup.py install
    else
        echo "Unsupported operating system: ${os}."
        exit 1
    fi
else
    echo "pytorch already installed. Skipping."
fi
cd ../


# Finally install beer.
python setup.py install
