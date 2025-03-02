#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import threading
from concurrent.futures import ThreadPoolExecutor

import percept.utils.troubleshoot as troubleshoot

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import Marker
from percept_interfaces.srv import PosesToVectors
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np

class PerceptionNode(Node):
    def __init__(self, node_name='perception_node', max_threads=5):
        super().__init__(node_name)
        
        # threading
        self.max_threads = int(max_threads)
        self.thread_pool = ThreadPoolExecutor(max_threads)

        # Buffer to store latest pointclouds
        self.pointcloud_buffer = {}
        self.pointcloud_buffer_lock = threading.Lock()

        # Buffer to store latest joint states
        self.joint_state_buffer = {}
        self.joint_state_buffer_lock = threading.Lock()
        self.num_joint_states = 0

        # Publisher for results
        self.primitives_publisher = self.create_publisher(
            PointCloud2, 
            '/primitives', 
            1
        )

        self.processing = False


    def run_pipeline(self, pointcloud_buffer, camera_tfs, agent_tfs, joint_states):
        try:
            if self.processing:
                return
            self.processing = True


            primitives_pos_result = self.pipeline.run_pipeline(
                pointcloud_buffer, camera_tfs, agent_tfs, joint_states)                    

            # if primitives_pos_result is not None and primitives_distance_vectors is not None:
            if primitives_pos_result is not None:
                # NOTE: temp fix
                primitives_pos_msg = self.make_pointcloud_msg(primitives_pos_result)

                # publish messages
                if primitives_pos_msg is not None:
                    self.primitives_publisher.publish(primitives_pos_msg)

            self.processing = False

        except Exception as e:
            self.get_logger().error(troubleshoot.get_error_text(e), exc_info=True)


    def make_pointcloud_msg(self, points_array):
        # Define header
        header = pc2.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"

        points_list = points_array

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_msg = pc2.create_cloud(header, fields, points_list)
        return point_cloud_msg
    
    def make_distance_vectors_visualization_msg(self, points_array):
        if len(points_array) == 0:
            return None
        
        marker = Marker()
        marker.header.frame_id = "agent_frame"  # Set to map frame TODO: change to agent frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "distvec_vis"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.03  # Line thickness
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8  # Alpha (transparency)
        origin = np.array([0.0, 0.0, 0.0])

        # create new points array with origin and endpoints (interleaved)
        tmp_points_array = np.empty((2 * len(points_array), 3), dtype=np.float32)
        tmp_points_array[0::2] = origin  # Even indices: origin
        tmp_points_array[1::2] = points_array  # Odd indices: endpoints

        # covert to list of geometry_msgs/Point msgs
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in tmp_points_array]
        
        return marker

    def shutdown(self):
        self.thread_pool.shutdown(wait=True)
        self.get_logger().info("Shutting down node.")