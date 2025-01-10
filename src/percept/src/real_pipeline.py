#!/usr/bin/env python3

import rospy

import argparse
import utils.troubleshoot as troubleshoot

from perception_pipeline import PerceptionPipeline
from perception_node import PerceptionNode

from sensor_msgs.msg import PointCloud2
from utils.camera_helpers import create_tf_matrix_from_euler


class RealPerceptionPipeline(PerceptionPipeline):
    def __init__(self, load_static_config=False):
        super().__init__()

        # load configs
        self.load_and_setup_pipeline_configs()

        if load_static_config:
            rospy.loginfo("loading static camera configs...")
            self.load_and_setup_static_camera_configs()

        # finish setup
        super().setup()

    def load_and_setup_pipeline_configs(self):
        self.perception_pipeline_config = rospy.get_param("perception_pipeline_config/", None)  
        self.scene_bounds = self.perception_pipeline_config['scene_bounds']
        self.cubic_size = self.perception_pipeline_config['voxel_props']['cubic_size']
        self.voxel_resolution = self.perception_pipeline_config['voxel_props']['voxel_resolution']

    def load_and_setup_static_camera_configs(self):
        self.static_camera_config = rospy.get_param("static_camera_config/", None)  

        def setup_static_cameras(static_camera_config):
            self.camera_names = list(static_camera_config.keys())
            self.tfs = dict()
            for camera_name in self.camera_names:
                self.tfs[camera_name] = create_tf_matrix_from_euler(static_camera_config[camera_name]['pose'])
            rospy.loginfo(f"static camera '{camera_name}' setup complete")
        
        setup_static_cameras(self.static_camera_config)


class RealPerceptionNode(PerceptionNode):
    def __init__(self, args):
        rospy.init_node('real_perception_node')
        super().__init__()

        # Initialize pipeline
        self.pipeline = RealPerceptionPipeline(
            load_static_config=args.static
        )
        self.setup_ros_subscribers()

    def setup_ros_subscribers(self):
        # Set up subscribers for each camera
        # right now we only have static camera callback
        self.subscribers = {}
        for camera_name in self.pipeline.camera_names:
            topic = f'/cameras/{camera_name}/depth/color/points'
            self.subscribers[camera_name] = rospy.Subscriber(
                topic, PointCloud2, self.static_camera_callback, callback_args=camera_name)

    def static_camera_callback(self, msg, camera_name):
        with self.buffer_lock:
            self.pointcloud_buffer[camera_name] = msg

            # Check if we have data from all cameras
            if len(self.pointcloud_buffer) == len(self.pipeline.camera_names):
                buffer_copy = self.pointcloud_buffer.copy()
                tfs = self.pipeline.tfs # use static camera tfs
                future = self.executor.submit(self.run_pipeline, buffer_copy, tfs, False)
                self.pointcloud_buffer.clear()

def main():
    parser = argparse.ArgumentParser(description="Configurable ROS Node")
    parser.add_argument('--static', action='store_true', help="Use static configuration instead of listening to a topic")
    args = parser.parse_args(rospy.myargv()[1:])  
    node = RealPerceptionNode(args)
    return node

if __name__ == "__main__":
    try:
        node = main()
        rospy.spin()
    except rospy.ROSInterruptException:
        node.shutdown()


