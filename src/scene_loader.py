#!/usr/bin/env python3


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

import yaml

class SceneCreator(Node):
    def __init__(self, node_name='scene_loader'):
        super().__init__(node_name)

        default_config_path = 'src/percept/assets/benchmark_scenes/obstacles1.yaml'
        self.declare_parameter('obstacles_config_path', default_config_path)
        config_path = self.get_parameter('obstacles_config_path').value

        # Publisher for obstacle positions
        self.obstacles_publisher = self.create_publisher(
            PointCloud2,
            '/primitives',
            1
        )

        # Load obstacles from yaml
        with open(config_path, "r") as f:
            self.obstacles_config = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Create timer for publishing
        self.create_timer(0.1, self.publish_obstacles)  # 10Hz

    def publish_obstacles(self):
        try:
            # Extract positions from obstacles config
            positions = []
            for obstacle in self.obstacles_config['obstacles']:
                positions.append(obstacle['position'])
            
            positions_array = np.array(positions, dtype=np.float32)
            
            # Create and publish pointcloud message
            obstacles_msg = self.make_pointcloud_msg(positions_array)
            self.obstacles_publisher.publish(obstacles_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing obstacles: {str(e)}')

    def make_pointcloud_msg(self, points_array):
        # Define header
        header = pc2.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_msg = pc2.create_cloud(header, fields, points_array)
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()