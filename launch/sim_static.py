import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction

from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import yaml
import os
from copy import deepcopy


def yaml_to_dict(path_to_yaml):
    with open(path_to_yaml, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)
    
def get_path(pkg_share:str, *paths):
    return os.path.join(pkg_share, *paths)


def create_sim_streamer_group(pkg_share:str, debug_mode, disable_render):
    
    
    sim_streamer_params = {
        'sim_config': get_path(pkg_share, 'config', 'sim_conf.yaml'),
        'env_id': 'SimEnv-v1',
        'debug_mode': debug_mode,
        'disable_render': disable_render
    }
    sim_streamer_group = [
        Node(
            package='percept',
            executable='sim_stream.py',
            name='sim_streamer',
            output='screen',
            parameters=[sim_streamer_params],
            namespace='sim_streamer'
        )
    ]
    return GroupAction(sim_streamer_group)



def generate_launch_description():
    pkg_share = get_package_share_directory('percept')
    
    # Declare the debug_mode launch argument
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug mode'
    )

    disable_render_arg = DeclareLaunchArgument(
        'disable_render',
        default_value='false',
        description='Disable rendering'
    )
    
    sim_streamer_group = create_sim_streamer_group(
        pkg_share,
        LaunchConfiguration('debug_mode'),
        LaunchConfiguration('disable_render')
    )

    # Combine all nodes into the launch description
    return LaunchDescription([
        debug_mode_arg,
        sim_streamer_group
    ])