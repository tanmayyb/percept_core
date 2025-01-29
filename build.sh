#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
conda activate percept_env
# source install/setup.bash
colcon clean workspace -y && colcon build --symlink-install
source install/setup.bash