#!/usr/bin/env python3

import rospy
# from message_filters import Subscriber, ApproximateTimeSynchronizer
import threading
from concurrent.futures import ThreadPoolExecutor

import argparse
import utils.troubleshoot as troubleshoot

from perception_pipeline import PerceptionPipeline
from utils.camera_helpers import create_tf_matrix_from_euler, create_tf_matrix_from_msg

from sensor_msgs.msg import Image
from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class SimPerceptionPipeline(PerceptionPipeline):
    def __init__(self):
        super().__init__()

        # load configs
        self.load_and_setup_pipeline_configs()
        self.load_and_setup_static_camera_configs()

        # finish setup
        super().setup()

    def load_and_setup_pipeline_configs(self):
        self.perception_pipeline_config = rospy.get_param("perception_pipeline_config/", None)  
        self.scene_bounds = self.perception_pipeline_config['scene_bounds']
        self.cubic_size = self.perception_pipeline_config['voxel_props']['cubic_size']
        self.voxel_resolution = self.perception_pipeline_config['voxel_props']['voxel_resolution']

    def load_and_setup_static_camera_configs(self):
        self.static_camera_config = rospy.get_param("/static_camera_config/", None)  

        def setup_cameras(static_camera_config):
            self.cameras = dict()
            for camera_name, camera_config in static_camera_config.items():
                self.cameras[camera_name] = dict()
                # tf_matrix = create_tf_matrix_from_euler(camera_config['pose'])
                # self.cameras[camera_name]['tf'] = tf_matrix
                # rospy.loginfo(f"camera '{camera_name}' setup complete")
            self.camera_names = list(self.cameras.keys())
        setup_cameras(self.static_camera_config)

    def create_observation(self, pointclouds:dict):
        obs = dict()
        for camera_name in self.camera_names:
            obs[camera_name] = dict()
            obs[camera_name]['pcd'] = pointclouds[camera_name]
            # obs[camera_name]['tf'] = self.cameras[camera_name]['tf']
        return obs

    def run_pipeline(self, pointclouds:dict):
        try:
            ret = self.create_observation(pointclouds)
            ret = self.run(ret, sim=True)
            return self.make_pcd_msg(ret)
        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))

    def make_pcd_msg(self, points_array):
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


class SimPerceptionNode:
    def __init__(self, max_threads=5):
        rospy.init_node('sim_perception_node')
        
        # threading
        self.max_threads = max_threads
        self.executor = ThreadPoolExecutor(max_threads)

        # Initialize pipeline
        self.pipeline = SimPerceptionPipeline()

        # Set up subscribers for each camera
        self.subscribers = {}
        for camera_name in self.pipeline.camera_names:
            topic = f'/cameras/{camera_name}/depth/color/points'
            self.subscribers[camera_name] = rospy.Subscriber(
                topic, PointCloud2, self.pointcloud_callback, callback_args=camera_name)

        # Buffer to store latest pointclouds
        self.pointcloud_buffer = {}
        self.buffer_lock = threading.Lock()

        # Publisher for results
        self.result_pub = rospy.Publisher('/primitives', PointCloud2, queue_size=10)

    def pointcloud_callback(self, msg, camera_name):
        with self.buffer_lock:
            self.pointcloud_buffer[camera_name] = msg

            # Check if we have data from all cameras
            if len(self.pointcloud_buffer) == len(self.pipeline.camera_names):
                # Process pointclouds
                # rospy.loginfo(f'{self.pointcloud_buffer.keys()}, {self.pipeline.camera_names}')
                buffer_copy = self.pointcloud_buffer.copy()
                future = self.executor.submit(self.process_pointclouds, buffer_copy)
                self.pointcloud_buffer.clear()

    def process_pointclouds(self, pointcloud_buffer):
        try:
            result = self.pipeline.run_pipeline(pointcloud_buffer)
            # rospy.loginfo(f'{result}')
            if result is not None:
                self.result_pub.publish(result)
        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))

    def shutdown(self):
        self.executor.shutdown(wait=True)
        rospy.loginfo("Shutting down node.")


def main():
    # parser = argparse.ArgumentParser(description="Configurable ROS Node")
    # parser.add_argument('--static', action='store_true', help="Use static configuration instead of listening to a topic")
    # args = parser.parse_args(rospy.myargv()[1:])  
    # node = SimPerceptionNode(args, max_threads=5)
    node = SimPerceptionNode(max_threads=5)

    return node

if __name__ == "__main__":
    try:
        node = main()
        rospy.spin()
    except rospy.ROSInterruptException:
        node.shutdown()