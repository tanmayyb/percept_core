#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import PointCloud2, PointField

import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros

# from pyrep import PyRep
# from pyrep.objects.vision_sensor import VisionSensor

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os
import argparse

from utils.troubleshoot import get_error_text
from utils.camera_helpers import create_tf_msg_from_xyzrpy

import gymnasium as gym
import mani_skill
import torch


class SimStreamer(Node):
    # Streams Scene from CoppeliaSim to ROS and back!
    def __init__(self):
        super().__init__('sim_streamer')

        self.env_name = "RoboCasaKitchen-v1"
        self.env = None

        self.init_sim()

    def init_sim(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        self.env = gym.make(self.env_name, obs_mode="state", control_mode="pd_joint_pos", render_mode="human")
        obs = self.env.reset()


    def run(self):
        while rclpy.ok():
            action = self.env.action_space.sample()
            _ = self.env.step(action)
            self.env.render() 
  


if __name__ == '__main__':
    rclpy.init()
    
    try:
        node = SimStreamer()
        node.run()
        # rclpy.spin(node)
    except KeyboardInterrupt:
        node.env.close()
        node.destroy_node()
        rclpy.shutdown()
