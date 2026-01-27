#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import time

class DynamicSceneLoader(Node):
    def __init__(self, node_name='dynamic_scene_loader'):
        super().__init__(node_name)
        
        # Parameters
        self.declare_parameter('filepath', '/tmp/dynamic_scene.npy')
        self.declare_parameter('frame_rate', 30.0) # Hz
        self.declare_parameter('ping_pong', False) # If True, plays forward then reverse

        self.filepath = self.get_parameter('filepath').get_parameter_value().string_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        self.ping_pong = self.get_parameter('ping_pong').get_parameter_value().bool_value

        self.obstacles_publisher = self.create_publisher(
            PointCloud2,
            '/pointcloud',
            10
        )

        # Load Data
        try:
            start_load = time.time()
            self.frames = np.load(self.filepath)
            end_load = time.time()
            
            # Validation
            if self.frames.ndim != 3:
                raise ValueError(f"Expected 3D array (frames, points, coords), got shape {self.frames.shape}")
                
            self.num_frames = self.frames.shape[0]
            self.get_logger().info(
                f"Loaded {self.num_frames} frames ({self.frames.shape[1]} points/frame) "
                f"in {end_load - start_load:.4f}s"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load scene: {e}")
            self.destroy_node()
            return

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Playback State
        self.current_idx = 0
        self.direction = 1 # 1 for forward, -1 for backward

        # Timer
        period = 1.0 / self.frame_rate
        self.create_timer(period, self.publish_next_frame)
        self.get_logger().info(f"Streaming started at {self.frame_rate}Hz")

    def publish_next_frame(self):
        try:
            # Get current frame data
            points = self.frames[self.current_idx]
            
            # Publish
            msg = self.make_pointcloud_msg(points)
            self.obstacles_publisher.publish(msg)

            # Update Index
            self.update_index()

        except Exception as e:
            self.get_logger().error(f'Error publishing frame {self.current_idx}: {str(e)}')

    def update_index(self):
        if self.ping_pong:
            # Reverse loop logic
            next_idx = self.current_idx + self.direction
            
            if next_idx >= self.num_frames:
                self.direction = -1
                self.current_idx = self.num_frames - 2 # Step back immediately
            elif next_idx < 0:
                self.direction = 1
                self.current_idx = 1 # Step forward immediately
            else:
                self.current_idx = next_idx
        else:
            # Standard forward loop
            self.current_idx = (self.current_idx + 1) % self.num_frames

    def make_pointcloud_msg(self, points_array):
        header = pc2.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "panda_link0"
        return pc2.create_cloud(header, self.fields, points_array)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicSceneLoader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()