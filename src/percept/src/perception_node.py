#!/usr/bin/env python3

import rospy

import threading
from concurrent.futures import ThreadPoolExecutor

import utils.troubleshoot as troubleshoot

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import numpy as np

class PerceptionNode:
    def __init__(self, max_threads=5):
        
        # threading
        self.max_threads = max_threads
        self.executor = ThreadPoolExecutor(max_threads)

        # Buffer to store latest pointclouds
        self.pointcloud_buffer = {}
        self.buffer_lock = threading.Lock()

        # Publisher for results
        self.primitives_publisher = rospy.Publisher('/primitives', PointCloud2, queue_size=10)

        # NOTE: temp fix
        # self.distance_vectors_publisher = rospy.Publisher('/distance_vectors', PointCloud2, queue_size=10)
        self.distance_vectors_visualization_publisher = rospy.Publisher('/distvec_vis', Marker, queue_size=10)

    def run_pipeline(self, pointcloud_buffer, tfs, use_sim):
        try:
            # result = self.pipeline.run_pipeline(
            #     pointcloud_buffer, tfs, use_sim=use_sim)
            # if result is not None:
            #     self.primitives_publisher.publish(self.make_pointcloud_msg(result))


            # NOTE: temp fix
            # TODO: add params for which results to publish
            # TODO: add params for which NOTE: visualizations to publish

            agent_config = rospy.get_param("agent_pos/", None)
            agent_pos = np.array([agent_config['x'], agent_config['y'], agent_config['z']])

            primitives_pos_result, primitives_distance_vectors = self.pipeline.run_pipeline(
                pointcloud_buffer, tfs, agent_pos, use_sim=use_sim)            
                
            # NOTE: temp fix
            primitives_pos_msg = self.make_pointcloud_msg(primitives_pos_result)
            primitives_distance_vectors_msg = self.make_distance_vectors_visualization_msg(primitives_distance_vectors)

            # publish messages
            if primitives_pos_msg is not None:
                self.primitives_publisher.publish(primitives_pos_msg)
            if primitives_distance_vectors_msg is not None:
                self.distance_vectors_visualization_publisher.publish(primitives_distance_vectors_msg)
        
        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))

    def make_pointcloud_msg(self, points_array):
        # Define header
        header = rospy.Header()
        header.stamp = rospy.Time.now()
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
        marker.header.stamp = rospy.Time.now()
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
        agent_config = rospy.get_param("agent_pos/", None)
        origin = np.array([agent_config['x'], agent_config['y'], agent_config['z']])

        # create new points array with origin and endpoints (interleaved)
        tmp_points_array = np.empty((2 * len(points_array), 3), dtype=np.float32)
        tmp_points_array[0::2] = origin  # Even indices: origin
        tmp_points_array[1::2] = points_array  # Odd indices: endpoints

        # covert to list of geometry_msgs/Point msgs
        marker.points = list(map(lambda p: Point(x=p[0], y=p[1], z=p[2]), tmp_points_array))

        return marker

    def shutdown(self):
        self.executor.shutdown(wait=True)
        rospy.loginfo("Shutting down node.")