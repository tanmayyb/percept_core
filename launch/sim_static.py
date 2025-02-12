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

# def create_perception_group(pkg_share:str):
#     real_pipeline_params = {
#         'static_camera_config': get_path(pkg_share, 'config', 'sim_static_cams.yaml'),
#         'sim_pipeline_config': get_path(pkg_share, 'config', 'sim_pipeline_conf.yaml'), 
#         'agent_config': get_path(pkg_share, 'config', 'agent_conf.yaml'),
#         'static_scene': True,
#         'static_agent': True
#     }
    
#     static_tf_publisher_params = {
#         'static_camera_config': get_path(pkg_share, 'config', 'sim_static_cams.yaml'),
#         'agent_config': get_path(pkg_share, 'config', 'agent_conf.yaml')
#     }

#     namespace = 'perception'
#     print(pkg_share)
#     # Define the perception group
#     perception_group = [
#             Node(
#                 package='percept',
#                 executable='sim_pipeline.py',
#                 name='sim_perception_node',
#                 output='screen',
#                 parameters=[sim_pipeline_params], # load all configs
#                 namespace=namespace
#             ),
#             Node(
#                 package='percept',
#                 executable='static_tf_publisher.py',
#                 name='static_tf_publisher',
#                 parameters=[static_tf_publisher_params],
#                 output='screen',
#                 namespace=namespace  
#             ),
#             Node(
#                 package='rviz2',
#                 executable='rviz2',
#                 name='perception_rviz',
#                 arguments=['-d', get_path(pkg_share, 'config', 'perception.rviz')],
#                 namespace='perception' 
#             )
#         ]

#     return GroupAction(perception_group)


# def create_realsense_group(pkg_share:str):
#     realsense_group = []
#     static_cams = yaml_to_dict(get_path(pkg_share, 'config', 'rs_static_cams.yaml'))
#     global_params = yaml_to_dict(get_path(pkg_share, 'config', 'rs_cameras.yaml'))
#     namespace = 'cameras'
#     for camera_id, camera_info in static_cams.items():  
#         params = deepcopy(global_params)
#         # params['camera_name'] = camera_id
#         # params['camera_namespace'] = namespace
#         params['serial_no'] = camera_info['serial_no']
#         node = Node(
#             package='realsense2_camera',
#             namespace=namespace,
#             name=camera_id,
#             executable='realsense2_camera_node',
#             parameters=[params],
#             # output=_output,
#             # arguments=['--ros-args', '--log-level', 'debug'],
#             # emulate_tty=True,
#         )
#         realsense_group.append(node)
#     return GroupAction(realsense_group)


def create_sim_streamer_group(pkg_share:str):
    sim_streamer_group = [
        Node(
            package='percept',
            executable='sim_stream.py',
            name='sim_streamer',
            output='screen',
            # parameters=[sim_streamer_params],
            namespace='sim_streamer'
        )
    ]
    return GroupAction(sim_streamer_group)



def generate_launch_description():
    pkg_share = get_package_share_directory('percept')
    # perception_group = create_perception_group(pkg_share)
    # realsense_group = create_realsense_group(pkg_share)

    sim_streamer_group = create_sim_streamer_group(pkg_share)

    
    # Combine all nodes into the launch description
    return LaunchDescription([
        # perception_group,
        # realsense_group,
        sim_streamer_group
    ])