#!/usr/bin/env python3

import rospy
import rospkg
from sensor_msgs.msg import PointCloud2, PointField

import sensor_msgs.point_cloud2 as pc2
import tf2_ros

from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os
import argparse

from utils.troubleshoot import get_error_text
from utils.camera_helpers import create_tf_msg_from_xyzrpy





class SimStreamer:
    # Streams Scene from CoppeliaSim to ROS and back!
    def __init__(self, args):
        rospy.init_node('sim_streamer')
        
        # setup
        self.load_and_setup_launch_configs()
        self.init_coppeliasim()
        self.config_coppeliasim()
              
        # Publishers
        self.set_publishers()
        self.setup_frames()
        
        # Publishing rate
        self.rate = rospy.Rate(30)  # 30 Hz


        # start
        self.pr.start()


    def load_and_setup_launch_configs(self):
        self.static_camera_config = rospy.get_param("static_camera_config/", None)  
        self.scene_config = rospy.get_param("scene_config/", None)  

        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('percept')

        self.no_headless = rospy.get_param("sim_streamer/no_headless/", False)  


    def init_coppeliasim(self, no_headless:bool=False):
        # get scene file path
        self.scene_file = os.path.join(self.pkg_path, self.scene_config['scene_file'])

        # start pyrep and load scene        
        self.pr = PyRep()
        self.pr.launch(self.scene_file, headless=not(self.no_headless))
        self.pr.set_simulation_timestep(self.scene_config['sim_timestep'])


    def config_coppeliasim(self): # init cameras from config file
        # need to add functionality later for scene
        self.cameras = dict()
        def setup_cameras(static_camera_config):
            # https://github.com/stepjam/PyRep/blob/8f420be8064b1970aae18a9cfbc978dfb15747ef/pyrep/objects/vision_sensor.py#L18
            for camera_name, camera_config in static_camera_config.items():
                cam_position = camera_config['pose']['position']
                cam_orientation = camera_config['pose']['orientation']
                try:
                    rospy.loginfo(f'Setting up camera: {camera_name} ({camera_config["nickname"]})')
                    self.cameras[camera_name] = VisionSensor.create(
                        camera_config['resolution'],
                        position=[cam_position['x'], cam_position['y'], cam_position['z']],
                        orientation=[cam_orientation['roll'], cam_orientation['pitch'], cam_orientation['yaw']]
                    )
                except Exception as e:
                    rospy.logerr(get_error_text(e))


            self.camera_names = list(self.cameras.keys())
        setup_cameras(self.static_camera_config)

    def setup_frames(self): # using config file
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        
        def create_tf_msgs(static_camera_config):
            static_transforms = []
            for camera_name, camera_config in static_camera_config.items():
                position = camera_config['pose']['position']
                orientation = camera_config['pose']['orientation']
                child_frame = f'{camera_name}_link'
                static_transforms.append(
                    create_tf_msg_from_xyzrpy(child_frame, 
                        position['x'], position['y'], position['z'],
                        orientation['roll'], orientation['pitch'], orientation['yaw']
                    )
                )
            return static_transforms

        msgs = create_tf_msgs(self.static_camera_config)
        self.tf_broadcaster.sendTransform(msgs)


    def set_publishers(self):
        self.publishers = dict()
        for camera_name in self.camera_names:
            # rospy.loginfo(f'Setting up publisher for {camera_name}')
            self.publishers[camera_name] = rospy.Publisher(
                f'/cameras/{camera_name}/depth/color/points', 
                PointCloud2, queue_size=10)

    def create_point_cloud_msg(self, points_array, camera_name):
        points = points_array.reshape(-1, 3)
       
        # Create point cloud message
        frame_id = 'map'
        # frame_id = f'{camera_name}_link'
        msg = PointCloud2()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        
        # Create point cloud from numpy array
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * float32 (4 bytes)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        
        return msg   

    def publish_pcds(self, pointclouds:dict):
        for camera_name, points in pointclouds.items():
            msg = self.create_point_cloud_msg(points, camera_name)
            self.publishers[camera_name].publish(msg)

    def capture_pcds(self, ):
        pointclouds = dict()
        for camera_name, camera in self.cameras.items():
            points = camera.capture_pointcloud()
            pointclouds[camera_name] = points
        return pointclouds

    def run(self):
        while not rospy.is_shutdown():           
            # Convert to point cloud
            pcds = self.capture_pcds()

            # Publish
            self.publish_pcds(pcds)

            # Step simulation
            self.pr.step()
            
            self.rate.sleep()
            
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
        rospy.loginfo("Shutting down SimStreamer node")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sim Streamer Node")
    args = parser.parse_args(rospy.myargv()[1:])  
    
    try:
        node = SimStreamer(args)
        node.run()
    except rospy.ROSInterruptException:
        node.shutdown()
