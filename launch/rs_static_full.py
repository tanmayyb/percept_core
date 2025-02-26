import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction

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

def create_perception_group(pkg_share:str, show_pipeline_delays, show_total_pipeline_delay):
    real_pipeline_params = {
        'static_camera_config': get_path(pkg_share, 'config', 'rs_static_cams.yaml'),
        'real_pipeline_config': get_path(pkg_share, 'config', 'rs_pipeline_conf.yaml'), 
        'agent_config': get_path(pkg_share, 'config', 'agent_conf.yaml'),
        'static_scene': True,
        'static_agent': True,
        'show_pipeline_delays': show_pipeline_delays,
        'show_total_pipeline_delay': show_total_pipeline_delay,
    }
    
    static_tf_publisher_params = {
        'static_camera_config': get_path(pkg_share, 'config', 'rs_static_cams.yaml'),
        'agent_config': get_path(pkg_share, 'config', 'agent_conf.yaml')
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
                arguments=['-d', get_path(pkg_share, 'rviz2', 'perception_full.rviz')],
                namespace='perception' 
            )
        ]

    return GroupAction(perception_group)


def create_realsense_group(pkg_share:str):
    realsense_group = []
    static_cams = yaml_to_dict(get_path(pkg_share, 'config', 'rs_static_cams.yaml'))
    global_params = yaml_to_dict(get_path(pkg_share, 'config', 'rs_cameras.yaml'))
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
    return GroupAction(realsense_group)


def setup_fields_computer(context):
    try:
        planner_mode = LaunchConfiguration('planner_mode').perform(context)
    except Exception:
        planner_mode = 'oriented_pointmass'

    remappings = []
    if planner_mode == 'manipulator':
        remappings = [
            ('/get_min_obstacle_distance', '/manipulator/get_min_obstacle_distance'),
            ('/get_random_heuristic_circforce', '/manipulator/get_random_heuristic_force'),
            ('/get_obstacle_heuristic_circforce', '/manipulator/get_obstacle_heuristic_force'),
            ('/get_goal_heuristic_circforce', '/manipulator/get_goal_heuristic_force'),
            ('/get_velocity_heuristic_circforce', '/manipulator/get_velocity_heuristic_force'),
            ('/get_goalobstacle_heuristic_circforce', '/manipulator/get_goalobstacle_heuristic_force'),
            ('/get_random_heuristic_circforce', '/manipulator/get_random_heuristic_force'),
        ]
    else:
        remappings = [
            ('/get_min_obstacle_distance', '/oriented_pointmass/get_min_obstacle_distance'),
            ('/get_random_heuristic_circforce', '/oriented_pointmass/get_random_heuristic_force'),
            ('/get_obstacle_heuristic_circforce', '/oriented_pointmass/get_obstacle_heuristic_force'),
            ('/get_goal_heuristic_circforce', '/oriented_pointmass/get_goal_heuristic_force'),
            ('/get_velocity_heuristic_circforce', '/oriented_pointmass/get_velocity_heuristic_force'),
            ('/get_goalobstacle_heuristic_circforce', '/oriented_pointmass/get_goalobstacle_heuristic_force'),
            ('/get_random_heuristic_circforce', '/oriented_pointmass/get_random_heuristic_force'),
        ]
    return [Node(
            package='percept',
            executable='fields_computer',
            name='fields_computer',
            output='screen',
            parameters=[{
                'k_cf_velocity': 0.01,
                'k_cf_obstacle': 0.01,
                'k_cf_goal': 0.001,
                'k_cf_goalobstacle': 0.001,
                'k_cf_random': 0.001,
                'agent_radius': 0.050,
                'mass_radius': 0.050,
                'max_allowable_force': 20.0,
                'detect_shell_rad': 100000.0,
                'publish_force_vector': False,
                'show_processing_delay': LaunchConfiguration('show_processing_delay'),
                'show_requests': LaunchConfiguration('show_requests'),
            }],
            remappings=remappings,
        ),
    ]


def generate_launch_description():
    show_processing_delay_arg = DeclareLaunchArgument(
        'show_processing_delay',
        default_value='False',
        description='Show processing delay information'
    )

    show_requests_arg = DeclareLaunchArgument(
        'show_requests',
        default_value='False',
        description='Show service request information'
    )

    planner_mode_arg = DeclareLaunchArgument(
        'planner_mode',
        default_value='oriented_pointmass',
        description='Planner Mode'
    )    

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
    pkg_share = get_package_share_directory('percept')
    perception_group = create_perception_group(
        pkg_share, 
        LaunchConfiguration('show_pipeline_delays'), 
        LaunchConfiguration('show_total_pipeline_delay')
    )
    realsense_group = create_realsense_group(pkg_share)
    opaque_fields_computer_setup = OpaqueFunction(function=setup_fields_computer)
    
    # Combine all nodes into the launch description
    return LaunchDescription([
        show_processing_delay_arg,
        show_requests_arg,
        show_pipeline_delays_arg,
        show_total_pipeline_delay_arg,
        perception_group,
        realsense_group,
        opaque_fields_computer_setup,
    ])