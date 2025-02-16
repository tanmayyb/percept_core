from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='percept',
            executable='service_tester.py',
            name='service_tester',
            output='screen'
        )
    ])
