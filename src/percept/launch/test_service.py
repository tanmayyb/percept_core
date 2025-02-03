from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='percept',
            executable='test_service.py',
            name='test_service',
            output='screen'
        )
    ])
