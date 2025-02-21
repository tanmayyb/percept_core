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
            'assets/benchmark_scenes',
            # 'hole_in_the_wall_easy.yaml'
            # 'hole_in_the_wall_medium.yaml'
            # 'hole_in_the_wall_hard.yaml'
            # 'maze_2d.yaml'
            'random_points_3d.yaml'
        ),
        description='Path to the obstacles configuration file'
    )

    return LaunchDescription([
        obstacles_config_arg,
        Node(
                package='percept',
                executable='scene_loader.py',
                name='scene_loader',
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
                # # # hole in the wall example- EASY - Success
                # 'k_circular_force': 0.00001,
                # 'agent_radius': 0.05,
                # 'mass_radius': 0.025,
                # 'max_allowable_force': 15.0,
                # 'detect_shell_rad': 100000.0,
                # 'publish_force_vector': False,
                # 'show_processing_delay': False,

                # # # hole in the wall example- Medium - Success
                'k_circular_force': 0.0001,
                'agent_radius': 0.05,
                'mass_radius': 0.025,
                'max_allowable_force': 20.0,
                'detect_shell_rad': 100000.0,
                'publish_force_vector': False,
                'show_processing_delay': True,
            }],
            remappings=[
                ('/get_obstacle_heuristic_circforce', '/oriented_pointmass/get_obstacle_heuristic_force'),
                ('/get_goal_heuristic_circforce', '/oriented_pointmass/get_goal_heuristic_force'),
                ('/get_velocity_heuristic_circforce', '/oriented_pointmass/get_velocity_heuristic_force'),
                ('/get_goalobstacle_heuristic_circforce', '/oriented_pointmass/get_goalobstacle_heuristic_force'),
                ('/get_random_heuristic_circforce', '/oriented_pointmass/get_random_heuristic_force'),

                # ('/get_obstacle_heuristic_circforce', '/manipulator/get_obstacle_heuristic_force'),
                # ('/get_goal_heuristic_circforce', '/manipulator/get_goal_heuristic_force'),
                # ('/get_velocity_heuristic_circforce', '/manipulator/get_velocity_heuristic_force'),
                # ('/get_goalobstacle_heuristic_circforce', '/manipulator/get_goalobstacle_heuristic_force'),
                # ('/get_random_heuristic_circforce', '/manipulator/get_random_heuristic_force'),
            ]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='perception_rviz',
            arguments=['-d', get_path(pkg_share, 'rviz2', 'planning.rviz')],
            namespace='perception' 
        )
    ])