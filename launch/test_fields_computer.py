from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def get_path(pkg_share:str, *paths):
    return os.path.join(pkg_share, *paths)

def generate_launch_description():
    pkg_share = get_package_share_directory('percept')
    obstacles_config_arg = DeclareLaunchArgument(
        'obstacles_config_path',
        default_value=os.path.join(
            pkg_share,
            'config',
            'obstacles3.yaml'
        ),
        description='Path to the obstacles configuration file'
    )

    return LaunchDescription([
        obstacles_config_arg,
        Node(
                package='percept',
                executable='scene_creator.py',
                name='scene_creator',
                parameters=[{
                    'obstacles_config_path': LaunchConfiguration('obstacles_config_path')
                }],
                output='screen'
            ),
        # Node(
        #     package='percept',
        #     executable='service_tester.py',
        #     name='service_tester',
        #     output='screen'
        # ),
        Node(
            package='percept',
            executable='fields_computer',
            name='fields_computer',
            output='screen',
            parameters=[{
                'k_circular_force': 0.0010,
                'agent_radius': 0.05,
                'mass_radius': 0.025,
                'max_allowable_force': 20.0,
                'detect_shell_rad': 1.0,
            }],
            remappings=[
                ('/get_heuristic_circforce', '/oriented_pointmass/get_obstacle_force'),
            ]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='perception_rviz',
            arguments=['-d', get_path(pkg_share, 'config', 'planning.rviz')],
            namespace='perception' 
        )
    ])