#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import struct

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

class DensePointCloudPublisher(Node):
    def __init__(self):
        super().__init__('fake_realsense')
        # Declare and get the topic parameter.
        self.declare_parameter('topic', '/cameras/camera_2/depth/color/points')
        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.publisher_ = self.create_publisher(PointCloud2, topic, 10)
        # Timer to publish at ~33 Hz.
        self.timer = self.create_timer(0.03, self.timer_callback)

        # Create a dense grid within a 0.5 x 0.5 x 1.0 volume.
        # For x and y, center about 0: from -0.25 to 0.25.
        # For z, from 0.0 to 1.0.
        spacing = 0.040
        x_vals = np.arange(-0.25, 0.25 + spacing/2, spacing)
        y_vals = np.arange(-0.25, 0.25 + spacing/2, spacing)
        z_vals = np.arange(0.0, 1.0 + spacing/2, spacing)

        # Create a meshgrid and stack into a (N, 3) array.
        xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T.astype(np.float32)
        self.num_points = points.shape[0]

        # Create a structured numpy array with fields for x, y, z, and rgb.
        cloud_array = np.zeros(self.num_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32)
        ])
        cloud_array['x'] = points[:, 0]
        cloud_array['y'] = points[:, 1]
        cloud_array['z'] = points[:, 2]

        # Precompute a constant white color in float representation.
        # 'rgb' is packed as a float from an unsigned int.
        rgb_int = (255 << 16) | (255 << 8) | 255  # white (0xFFFFFF)
        self.rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
        cloud_array['rgb'] = self.rgb_float

        # Convert the numpy structured array to a bytes object.
        self.cloud_bytes = cloud_array.tobytes()

    def timer_callback(self):
        # Create and populate the PointCloud2 message.
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_2_link"  # Change if necessary.
        msg.height = 1      # Unorganized point cloud.
        msg.width = self.num_points
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16  # 4 fields x 4 bytes each.
        msg.row_step = msg.point_step * self.num_points
        msg.is_dense = True
        msg.data = self.cloud_bytes

        # Publish the message.
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published dense pointcloud with {self.num_points} points.')

def main(args=None):
    rclpy.init(args=args)
    node = DensePointCloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
