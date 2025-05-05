from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_share_directory
import os
from launch.conditions import IfCondition

def get_path(pkg_share:str, *paths):
    return os.path.join(pkg_share, *paths)

def setup_fields_computer(context):
    try:
        planner_mode = LaunchConfiguration('planner_mode').perform(context)
    except Exception:
        planner_mode = 'oriented_pointmass'

    use_cpu = LaunchConfiguration('use_cpu').perform(context)
    if use_cpu.lower() == 'true':
        node_executable = 'fields_computer_cpu'
    else:
        node_executable = 'fields_computer'

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
            ('/get_apf_heuristic_force', '/manipulator/get_apf_heuristic_force'),
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
            ('/get_apf_heuristic_force', '/oriented_pointmass/get_apf_heuristic_force'),
        ]
    return [Node(
            package='percept',
            executable=node_executable,
            name=node_executable,
            output='screen',
            parameters=[{
                'show_processing_delay': LaunchConfiguration('show_processing_delay'),
                'show_requests': LaunchConfiguration('show_requests'),
            }],
            remappings=remappings,
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('percept')
    obstacles_config_arg = DeclareLaunchArgument(
        'obstacles_config_path',
        default_value=os.path.join(
            pkg_share,
            'assets/benchmark_scenes',
            'auto_generated_scene.yaml'
        ),
        description='Path to the obstacles configuration file'
    )

    use_cpu_arg = DeclareLaunchArgument(
        'use_cpu',
        default_value='False',
        description='Use CPU implementation'
    )

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

    show_rviz_arg = DeclareLaunchArgument(
        'show_rviz',
        default_value='True',
        description='Show RVIZ'
    )    


    opaque_fields_computer_setup = OpaqueFunction(
        function=setup_fields_computer
    )
 
    return LaunchDescription([
        obstacles_config_arg,
        use_cpu_arg,
        show_processing_delay_arg,
        show_requests_arg,
        planner_mode_arg,
        Node(
                package='percept',
                executable='scene_loader.py',
                name='scene_loader',
                parameters=[{
                    'obstacles_config_path': LaunchConfiguration('obstacles_config_path'),
                    'publish_once': False
                }],
                output='screen'
            ),
        # Node(
        #     package='percept',
        #     executable='service_tester.py',
        #     name='service_tester',
        #     output='screen'
        # ),
        opaque_fields_computer_setup,
        show_rviz_arg,
        Node(
            package='rviz2',
            executable='rviz2',
            name='perception_rviz',
            arguments=['-d', get_path(pkg_share, 'rviz2', 'planning.rviz')],
            namespace='perception',
            condition=IfCondition(LaunchConfiguration('show_rviz'))
        )
    ])