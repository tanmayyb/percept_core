#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import threading
from concurrent.futures import ThreadPoolExecutor

import percept.utils.troubleshoot as troubleshoot
from percept.utils.config_helpers import load_yaml_as_dict

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
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
        self.buffer_lock = threading.Lock()

        # Publisher for results
        self.primitives_publisher = self.create_publisher(
            PointCloud2, 
            '/primitives', 
            10
        )

        # NOTE: temp fix
        # self.distance_vectors_publisher = self.create_publisher(PointCloud2, '/distance_vectors', 10)
        # self.distance_vectors_visualization_publisher = self.create_publisher(
        #     Marker,
        #     '/distvec_vis',
        #     10
        # )

    def run_pipeline(self, pointcloud_buffer, tfs, agent_pos, use_sim):
        try:
            # result = self.pipeline.run_pipeline(
            #     pointcloud_buffer, tfs, use_sim=use_sim)
            # if result is not None:
            #     self.primitives_publisher.publish(self.make_pointcloud_msg(result))

            # NOTE: temp fix
            # TODO: add params for which results to publish
            # TODO: add params for which NOTE: visualizations to publish

            # agent_config = self.get_parameter("agent_pos").get_parameter_value().string_value
            # agent_pos = np.array([agent_config['x'], agent_config['y'], agent_config['z']])

            # primitives_pos_result, primitives_distance_vectors = self.pipeline.run_pipeline(
            #     pointcloud_buffer, tfs, agent_pos, use_sim=use_sim)            

            primitives_pos_result = self.pipeline.run_pipeline(
                pointcloud_buffer, tfs, agent_pos, use_sim=use_sim)                    

            # if primitives_pos_result is not None and primitives_distance_vectors is not None:
            if primitives_pos_result is not None:
                # NOTE: temp fix
                primitives_pos_msg = self.make_pointcloud_msg(primitives_pos_result)
                # primitives_distance_vectors_msg = self.make_distance_vectors_visualization_msg(primitives_distance_vectors)

                # publish messages
                if primitives_pos_msg is not None:
                    self.primitives_publisher.publish(primitives_pos_msg)
                # if primitives_distance_vectors_msg is not None:
                #     self.distance_vectors_visualization_publisher.publish(primitives_distance_vectors_msg)
            
        except Exception as e:
            self.get_logger().error(troubleshoot.get_error_text(e))

    def make_pointcloud_msg(self, points_array):
        # Define header
        header = pc2.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

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

        # NOTE: temp fix
        # TODO: add agent pos in future!
        # agent_config = self.get_parameter("agent_pose").get_parameter_value().string_value
        # origin = np.array([agent_config['x'], agent_config['y'], agent_config['z']])
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

    def load_and_setup_static_agent_config(self, config_path):
        self.agent_config = load_yaml_as_dict(self, config_path)
        agent_pose = self.agent_config['agent_pose']
        self.agent_pos = np.array([agent_pose['x'], agent_pose['y'], agent_pose['z']])
        self.get_logger().info(f"static agent config loaded: {self.agent_pos}")