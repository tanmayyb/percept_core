from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def create_perception_group(context, pkg_share: str, obstacles_config_path, show_pipeline_delays, show_total_pipeline_delay):
    real_pipeline_params = {
        'enable_dynamic_cameras': False,
        'enable_dynamic_agents': False,
        'enable_robot_body_subtraction': False,
        'show_pipeline_delays': show_pipeline_delays.perform(context).lower() == 'true',
        'show_total_pipeline_delay': show_total_pipeline_delay.perform(context).lower() == 'true',
    }

    namespace = 'perception'
    obstacles_path = obstacles_config_path.perform(context)
    
    perception_group = [
        # Node(
        #     package='percept',
        #     executable='real_pipeline.py',
        #     name='real_perception_node',
        #     output='screen',
        #     parameters=[real_pipeline_params],
        #     namespace=namespace
        # ),
        Node(
            package='percept',
            executable='scene_loader.py',
            name='scene_loader',
            parameters=[{
                'obstacles_config_path': obstacles_path,
                'publish_once': False
            }],
            output='screen'
        ),
    ]

    return perception_group


def create_realsense_group(pkg_share: str):
    """Create the RealSense camera group."""
    # Add your RealSense camera nodes here if needed
    # For now, returning an empty list to maintain functionality
    return []


def setup_fields_computer(context):
    planner_mode = 'manipulator'
    node_executable = 'fields_computer'

    remap_keys = [
        ('get_min_obstacle_distance', 'get_min_obstacle_distance'),
        ('get_random_heuristic_circforce', 'get_random_heuristic_force'),
        ('get_obstacle_heuristic_circforce', 'get_obstacle_heuristic_force'),
        ('get_goal_heuristic_circforce', 'get_goal_heuristic_force'),
        ('get_velocity_heuristic_circforce', 'get_velocity_heuristic_force'),
        ('get_goalobstacle_heuristic_circforce', 'get_goalobstacle_heuristic_force'),
        ('get_apf_heuristic_circforce', 'get_apf_heuristic_force'),
        ('get_navigation_function_circforce', 'get_navigation_function_force'),
    ]
    
    remappings = []
    prefix = f"/{planner_mode}/"
    for src, dst in remap_keys:
        remappings.append((f"/{src}", f"{prefix}{dst}"))

    return [Node(
        package='percept',
        executable=node_executable,
        name=node_executable,
        output='screen',
        parameters=[{
            'show_processing_delay': LaunchConfiguration('show_processing_delay').perform(context).lower() == 'true',
            'show_requests': LaunchConfiguration('show_requests').perform(context).lower() == 'true',
        }],
        remappings=remappings,
    )]


def generate_launch_description():
    pkg_share = get_package_share_directory('percept')
    
    # Launch arguments
    args = [
        DeclareLaunchArgument(
            'obstacles_config_path',
            default_value=os.path.join(
                pkg_share,
                'assets/benchmark_scenes',
                'auto_generated_scene.yaml'
            ),
            description='Path to the obstacles configuration file'
        ),
        DeclareLaunchArgument(
            'show_processing_delay',
            default_value='False',
            description='Show processing delay information'
        ),
        DeclareLaunchArgument(
            'show_requests',
            default_value='False',
            description='Show service request information'
        ),
        DeclareLaunchArgument(
            'show_pipeline_delays',
            default_value='False',
            description='Show pipeline delays'
        ),
        DeclareLaunchArgument(
            'show_total_pipeline_delay',
            default_value='False',
            description='Show total pipeline delay'
        ),
    ]

    # Create groups and nodes
    opaque_perception_group = OpaqueFunction(
        function=lambda context: create_perception_group(
            context,
            pkg_share, 
            LaunchConfiguration('obstacles_config_path'),
            LaunchConfiguration('show_pipeline_delays'), 
            LaunchConfiguration('show_total_pipeline_delay')
        )
    )
    
    realsense_group = create_realsense_group(pkg_share)
    opaque_fields_computer_setup = OpaqueFunction(function=setup_fields_computer)
    
    return LaunchDescription([
        *args,
        opaque_perception_group,
        *realsense_group,
        opaque_fields_computer_setup,
    ])