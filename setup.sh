#!/bin/bash

# apt packages
sudo apt install python3-virtualenv

# env
if ! grep -q "export PERCEPT_ROOT=" ~/.bashrc || ! grep -q "export PERCEPT_ROOT=$(pwd)" ~/.bashrc; then
    sed -i '/export PERCEPT_ROOT/d' ~/.bashrc  # Remove existing PERCEPT_ROOT if present
    echo "export PERCEPT_ROOT=$(pwd)" >> ~/.bashrc
    echo "PERCEPT_ROOT updated in .bashrc to: $(pwd)"
fi

# python env
virtualenv -p $(which python3.8) --system-site-packages peract_env 
source ./peract_env/bin/activate
pip install --upgrade pip

# python-dependencies
pip install -r requirements.txt
