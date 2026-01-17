#!/usr/bin/env python3


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

import yaml, time
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

class SceneCreator(Node):
	def __init__(self, node_name='static_scene_loader'):
		super().__init__(node_name)
		
		self.pkg_name = 'percept_core'

		# self.loop_disable = True

		# self.publish_rate = 0.03
		self.declare_parameter('loop_disable', False)
		self.declare_parameter('publish_rate', 0.03)

		self.loop_disable = self.get_parameter('loop_disable').get_parameter_value().bool_value
		self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
	

		filepath = Path(get_package_share_directory(self.pkg_name)) / 'auto_generated_scene.npy'

		self.obstacles_publisher = self.create_publisher(
			PointCloud2,
			'/pointcloud',
			10
		)

		start = time.time()

		with open(filepath, "r") as f:
			self.positions_array = np.load(filepath)

		end = time.time()
		
		self.get_logger().info(f"Loaded {len(self.positions_array)} obstacles in {end - start} seconds")

		self.fields = [
			PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
			PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
			PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
		]

		# Create timer for publishing
		if self.loop_disable:
			self.publish_obstacles()
		else:
			self.create_timer(self.publish_rate, self.publish_obstacles)  # 10Hz

	def publish_obstacles(self):
		try:
			obstacles_msg = self.make_pointcloud_msg(self.positions_array)

			self.obstacles_publisher.publish(obstacles_msg)

		except Exception as e:
			self.get_logger().error(f'Error publishing obstacles: {str(e)}')

	def make_pointcloud_msg(self, points_array):
		# Define header
		header = pc2.Header()
		header.stamp = self.get_clock().now().to_msg()
		header.frame_id = "world"

		point_cloud_msg = pc2.create_cloud(header, self.fields, points_array)
		return point_cloud_msg

def main(args=None):
	rclpy.init(args=args)
	node = SceneCreator()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()

if __name__ == '__main__':
	main()