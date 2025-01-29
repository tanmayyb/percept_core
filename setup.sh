#!/bin/bash

# setup env variable
if ! grep -q "export PERCEPT_ROOT=" ~/.bashrc || ! grep -q "export PERCEPT_ROOT=$(pwd)" ~/.bashrc; then
    sed -i '/export PERCEPT_ROOT/d' ~/.bashrc  # Remove existing PERCEPT_ROOT if present
    echo "export PERCEPT_ROOT=$(pwd)" >> ~/.bashrc
    echo "PERCEPT_ROOT updated in .bashrc to: $(pwd)"
fi

# conda env
echo "Creating conda environment..."
conda create -n percept_env python=3.10 -y
eval "$(conda shell.bash hook)"
conda init bash
source ~/.bashrc
conda activate percept_env || { echo "Failed to activate conda environment"; exit 1; }
pip install --upgrade pip

# install CUDA packages
echo "Installing CUDA packages"
conda install -y -c conda-forge cupy numba

# python-dependencies
echo "Installing python-dependencies..."
pip install -r requirements.txt

# setup PyRep - for coppelia simulation
# mkdir libs
# cd libs
# git clone https://github.com/stepjam/pyrep.git
# cd pyrep
# pip install -r requirements.txt
# pip install .
