# Perception Pipeline

Goal: Fast (~30Hz) perception pipline for generating control parameters and workspace approximation in tight-constrained scenes for heuristic-based multi-agent planning framework (PMAF) by Laha et al 2023.

## Requirements

- Ubuntu 20.04 Focal Fossa ([iso](https://releases.ubuntu.com/focal/https:/))
- ROS 1 Noetic ([debian](http://wiki.ros.org/noetic/Installation/Debianhttps:/))


| Packages                                    | Links                                                                                          |
| :-------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| librealsenseSDK                             | [debian](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) |
| CoppeliaSim 4.1.0 EDU (for simulation) | [binaries](https://www.coppeliarobotics.com/previousVersionshttps:/)                           |

- URDF model of selected robot (for Robot-Body Subtraction from scene). We use Franka Emika Robots for Sim and Real-World tests.
-

## Setup

- Install aforementioned requirements and clone this git repo
- Run `./setup.sh` - this will create a virtual environment and install all dependencies and libraries
- Set `$COPPELIASIM_ROOT` to the CoppeliaSim installation directory (for simulation)

## Credits

### Collaborators

- Riddhiman Laha
- Tinayu Ren

### Projects

- CuPoch
- Open3D
- CoppeliaSim
