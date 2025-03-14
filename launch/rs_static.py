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

def create_perception_group(pkg_share:str, enable_robot_body_subtraction, show_pipeline_delays, show_total_pipeline_delay):
    real_pipeline_params = {
        'enable_dynamic_cameras': False,
        'enable_dynamic_agents': False,
        'enable_robot_body_subtraction': enable_robot_body_subtraction,
        'show_pipeline_delays': show_pipeline_delays,
        'show_total_pipeline_delay': show_total_pipeline_delay,
    }
    
    static_tf_publisher_params = {
        'static_camera_config': get_path(pkg_share, 'config', 'static_cameras_setup.yaml'),
        'static_agent_config': get_path(pkg_share, 'config', 'static_agents_setup.yaml')
    }

    namespace = 'perception'
    print(pkg_share)
    # Define the perception group
    perception_group = [
            Node(
                package='percept',
                executable='real_pipeline.py',
                name='real_perception_node',
                output='screen',
                parameters=[real_pipeline_params], # load all configs
                namespace=namespace
            ),
            Node(
                package='percept',
                executable='static_tf_publisher.py',
                name='static_tf_publisher',
                parameters=[static_tf_publisher_params],
                output='screen',
                namespace=namespace  
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='perception_rviz',
                arguments=['-d', get_path(pkg_share, 'rviz2', 'perception.rviz')],
                namespace='perception' 
            )
        ]

    return GroupAction(perception_group)


def create_realsense_group(pkg_share:str):
    realsense_group = []
    static_cams = yaml_to_dict(get_path(pkg_share, 'config', 'static_cameras_setup.yaml'))
    global_params = yaml_to_dict(get_path(pkg_share, 'config', 'rs_settings.yaml'))
    namespace = 'cameras'

    for camera_id, camera_info in static_cams.items():  
        params = deepcopy(global_params)
        # params['camera_name'] = camera_id
        # params['camera_namespace'] = namespace
        params['serial_no'] = camera_info['serial_no']
        node = Node(
            package='realsense2_camera',
            namespace=namespace,
            name=camera_id,
            executable='realsense2_camera_node',
            parameters=[params],
            # output=_output,
            # arguments=['--ros-args', '--log-level', 'debug'],
            # emulate_tty=True,
        )
        realsense_group.append(node)

    # camera_id, camera_info = list(static_cams.items())[0]   
    # params = deepcopy(global_params)
    # # params['camera_name'] = camera_id
    # # params['camera_namespace'] = namespace
    # params['serial_no'] = camera_info['serial_no']
    # node = Node(
    #     package='realsense2_camera',
    #     namespace=namespace,
    #     name=camera_id,
    #     executable='realsense2_camera_node',
    #     parameters=[params],
    #     # output=_output,
    #     # arguments=['--ros-args', '--log-level', 'debug'],
    #     # emulate_tty=True,
    # )
    # realsense_group.append(node)


    return GroupAction(realsense_group)


def generate_launch_description():
    show_pipeline_delays_arg = DeclareLaunchArgument(
        'show_pipeline_delays',
        default_value='false',
        description='Show pipeline delays'
    )
    show_total_pipeline_delay_arg = DeclareLaunchArgument(
        'show_total_pipeline_delay',
        default_value='false',
        description='Show total pipeline delay'
    )
    enable_robot_body_subtraction_arg = DeclareLaunchArgument(
        'enable_robot_body_subtraction',
        default_value='false',
        description='Enable robot body subtraction'
    )

    pkg_share = get_package_share_directory('percept')
    perception_group = create_perception_group(
        pkg_share, 
        LaunchConfiguration('enable_robot_body_subtraction'), 
        LaunchConfiguration('show_pipeline_delays'), 
        LaunchConfiguration('show_total_pipeline_delay')
    )
    realsense_group = create_realsense_group(pkg_share)
    
    # Combine all nodes into the launch description
    return LaunchDescription([
        show_pipeline_delays_arg,
        show_total_pipeline_delay_arg,
        enable_robot_body_subtraction_arg,
        perception_group,
        realsense_group,
    ])