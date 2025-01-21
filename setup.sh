#!/bin/bash

# setup env variable
if ! grep -q "export PERCEPT_ROOT=" ~/.bashrc || ! grep -q "export PERCEPT_ROOT=$(pwd)" ~/.bashrc; then
    sed -i '/export PERCEPT_ROOT/d' ~/.bashrc  # Remove existing PERCEPT_ROOT if present
    echo "export PERCEPT_ROOT=$(pwd)" >> ~/.bashrc
    echo "PERCEPT_ROOT updated in .bashrc to: $(pwd)"
fi

# conda env
conda create -n percept_env python=3.8
conda activate percept_env
pip install --upgrade pip

# python-dependencies
pip install -r requirements.txt

# setup PyRep - for coppelia simulation
mkdir libs
cd libs
git clone https://github.com/stepjam/pyrep.git
cd pyrep
pip install -r requirements.txt
pip install .