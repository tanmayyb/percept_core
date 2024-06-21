# do not include LD_LIBRARY_PATH and QT_QPA_PLATFORM_PLUGIN_PATH in the ~/.bashrc (RViz will conflict)
# they can be exported in the process when needed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export SETUPTOOLS_USE_DISTUTILS=stdlib
cd $COPPELIASIM_ROOT && ./coppeliaSim.sh -h $PMAF_ROOT/src/bimanual_planning_ros/vrep_scenes/dual_arms.ttt

roslaunch bimanual_planning_ros vrep_interface_dual_arms.launch task_sequence:=dual_arms_static1
roslaunch bimanual_planning_ros planning_moveit_dual_arms.launch