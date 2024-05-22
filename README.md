# percept

![alt text](imgs/banner.png)


## Debug Fixes

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