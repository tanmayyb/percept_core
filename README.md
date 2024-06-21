# percept

![alt text](imgs/banner.png)




```
export SETUPTOOLS_USE_DISTUTILS=stdlib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

cd $PERACT_ROOT
CUDA_VISIBLE_DEVICES=0 python eval.py \
    rlbench.tasks=[open_drawer] \
    rlbench.task_name='multi' \
    rlbench.demo_path=$PERACT_ROOT/data/val \
    framework.gpu=0 \
    framework.logdir=$PERACT_ROOT/ckpts/ \
    framework.start_seed=0 \
    framework.eval_envs=1 \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=10 \
    framework.csv_logging=True \
    framework.tensorboard_logging=True \
    framework.eval_type='last' \
    rlbench.headless=False

```



```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
cd $COPPELIASIM_ROOT && ./coppeliaSim.sh -h $PMAF_ROOT/src/bimanual_planning_ros/vrep_scenes/dual_arms.ttt

roslaunch bimanual_planning_ros vrep_interface_dual_arms.launch task_sequence:=dual_arms_static1
roslaunch bimanual_planning_ros planning_moveit_dual_arms.launch
```


## Debug Fixes (These Issues pop up on Non-Recommended Configs)

### OmegaConf Error #1
```
pkg_resources.extern.packaging.requirements.InvalidRequirement: .* suffix can only be used with `==` or `!=` operators
    PyYAML (>=5.1.*)
```
is related to the omegaconf package's METADATA file and NOT pyyaml. The authors of the omegaconf package stopped maintaining it after version 2.0.6 and this is a PEP-related bug (non-standard dependency specifier) that arises when a description generator (?) is invoked. Luckily there is an easy workaround.

`<NAME-OF-YOUR-PERACT-CONDA-ENVIRONMENT>` = peract
1. Navigate to package site:
```
cd ~/miniconda3/envs/<NAME-OF-YOUR-PERACT-CONDA-ENVIRONMENT>/lib/python3.8/site-packages/omegaconf-2.0.6.dist-info
```
2. Edit the METADATA file and change:

```
Requires-Dist: PyYAML (>=5.1.*)
to
Requires-Dist: PyYAML (>=5.1)
```

### Installation of ROS2 Plugin for Coppelia Sim (simROS2)

1. CoppeliaSim player [[link](https://www.coppeliarobotics.com/previousVersions)]
2. Plugin install [[link](https://manual.coppeliarobotics.com/en/ros2Tutorial.htm)], [[link](https://github.com/CoppeliaRobotics/simROS2)]
3. Include programming/include [[link](https://github.com/CoppeliaRobotics/include/tree/master)] [[link](https://manual.coppeliarobotics.com/)] 
4. Stubs generator [[link](https://github.com/CoppeliaRobotics/include/blob/master/simStubsGen/README.md)] dependencies 
5. Making sure all git repos had tag 4.5.1 and tweaking around the simROS2 CMakerLists.txt 


### Extra
```
git config --global user.name "John Doe"
git config --global user.email "johndoe@email.com"
```