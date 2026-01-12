from launch import LaunchDescription
from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import yaml, os
from copy import deepcopy


def generate_launch_description():
	package_name = 'percept_core'


	return LaunchDescription([
		Node(
			package=package_name,
			executable='static_scene_loader.py',
			name='static_scene_loader'
		)
	])