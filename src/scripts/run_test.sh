#! /bin/bash

cd $PERCEPT_ROOT
source peract_env/bin/activate

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export SETUPTOOLS_USE_DISTUTILS=stdlib

cd $PERCEPT_ROOT
python scripts/test_scene.py