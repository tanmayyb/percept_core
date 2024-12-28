#!/usr/bin/env python3

import rospy
import threading
from concurrent.futures import ThreadPoolExecutor
from message_filters import Subscriber, ApproximateTimeSynchronizer

from sensor_msgs.msg import Image
from std_msgs.msg import String

import numpy as np
import yaml
from perception_pipeline import PerceptionPipeline
import os

class RealTimePerceptionPipeline(PerceptionPipeline):
    def __init__(self):
        super().__init__()

        # load configs
        self.load_configs()


    def load_configs(self):
        # get config file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file_path = os.path.join(current_dir, "../config/cameras.yaml") # roslaunch param
        config_file_path = os.path.abspath(config_file_path)

        # load config
        try:
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)
                rospy.loginfo(f"Successfully loaded config: {config_file_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load config file: {e}")
            config = {}
        self.config = config

    def create_observation(self, msg):
        # read message and create observation
        image_array = np.frombuffer(msg.data, np.uint8)
        width = msg.width
        height = msg.height
        step = msg.step
        encoding = msg.encoding


        # self.observations = [
        #     {
        #         # Camera data for each camera (cam1, cam2, cam3)
        #         "cam1": {
        #             "rgb": array(...),        # RGB image data
        #             "depth": array(...),      # Depth image data
        #             "position": array(...),   # Camera position
        #             "resolution": array(...), # Camera resolution
        #             "extrinsics": array(...), # Camera extrinsic matrix
        #             "intrinsics": array(...), # Camera intrinsic matrix
        #             "pointcloud": array(...), # Point cloud data
        #         },
        #         "cam2": { ... },
        #         "cam3": { ... },

        #         # Robot arm data for each arm (panda0, panda1)
        #         "panda0": {
        #             "joint_pos": array(...),  # Joint positions
        #             "global_pos": array(...), # Global position
        #             "global_ang": array(...), # Global orientation
        #         },
        #         "panda1": { ... }
        #     },
        #     # Next timestep...
        # ]
        



    def run(self, msg): # for single camera
        # Goal:
        # run pipeline using new frames (from camera) and fixed extrinsics (from config file)

        # config file loaded in init: DONE

        # create observation when new message received 
        self.obs = dict()
        self.create_observation()
        # self.do_point_cloud_registration()

        # ...later steps



        # use config files/extrinsics and 
        # run pipeline using new frames (from camera) and fixed extrinsics (from config file)


        return None


class PerceptionNode:
    def __init__(self, max_threads=5):
        rospy.init_node('perception_node')

        # threading
        self.max_threads = max_threads
        self.executor = ThreadPoolExecutor(max_threads)
        self.lock = threading.Lock()

        # Publisher
        self.publisher = rospy.Publisher('/output_topic', String, queue_size=10)

        # Subscriber
        rospy.Subscriber('/Depth_Image', Image, self.callback)

        # # Subscribers
        # self.sub1 = Subscriber('/camera1/Depth_Image', Image)
        # self.sub2 = Subscriber('/camera2/Depth_Image', Image)
        # self.sub3 = Subscriber('/camera3/Depth_Image', Image)

        # # Synchronizer
        # self.sync = ApproximateTimeSynchronizer([self.sub1, self.sub2, self.sub3], queue_size=10, slop=0.1)
        # self.sync.registerCallback(self.callback)


        # Perception Pipeline
        self.perception_pipeline = RealTimePerceptionPipeline()
        
    def callback(self, msg):
        # Launch a thread to process the data
        self.executor.submit(self.process_and_publish, msg)

    # def synchronized_callback(self, img1, img2, img3):
    #         rospy.loginfo("Synchronized images received.")
            
    #         # Submit the synchronized processing to a thread
    #         self.executor.submit(self.process_and_publish, img1, img2, img3)

    def process_and_publish(self, msg):
        result = self.perception_pipeline.run(msg)

        # Publish the result
        # with self.lock:
        #     self.publisher.publish(result)
        #     rospy.loginfo(f"Published: {result}")

    def shutdown(self):
        # Shutdown the executor cleanly
        self.executor.shutdown(wait=True)
        rospy.loginfo("Shutting down node.")

if __name__ == "__main__":
    try:
        node = PerceptionNode(max_threads=5)
        rospy.spin()
    except rospy.ROSInterruptException:
        node.shutdown()
