#!/usr/bin/env python3

import rospy

import threading
from concurrent.futures import ThreadPoolExecutor

import utils.troubleshoot as troubleshoot

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class PerceptionNode:
    def __init__(self, max_threads=5):
        
        # threading
        self.max_threads = max_threads
        self.executor = ThreadPoolExecutor(max_threads)

        # Buffer to store latest pointclouds
        self.pointcloud_buffer = {}
        self.buffer_lock = threading.Lock()

        # Publisher for results
        self.publisher = rospy.Publisher('/primitives', PointCloud2, queue_size=10)

    def run_pipeline(self, pointcloud_buffer, tfs, use_sim):
        try:
            result = self.pipeline.run_pipeline(
                pointcloud_buffer, tfs, use_sim=use_sim)
            if result is not None:
                self.publisher.publish(self.make_pointcloud_msg(result))
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
    
    def shutdown(self):
        self.executor.shutdown(wait=True)
        rospy.loginfo("Shutting down node.")