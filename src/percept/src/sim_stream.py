#!/usr/bin/env python3

import rospy
import rospkg

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
import os

class SimStreamer:
    def __init__(self):
        rospy.init_node('sim_streamer')
        
        # Initialize CoppeliaSim
        rospack = rospkg.RosPack()
        SCENE_FILE = os.path.join(rospack.get_path('percept'), 'assets/scenes/default.ttt')
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=True)
        self.pr.start()
        
        # Get vision sensor
        self.vision_sensor = VisionSensor('Vision_sensor')
        
        # Publisher
        self.pub = rospy.Publisher('/cameras/camera_1/depth/color/points', PointCloud2, queue_size=10)
        
        # Publishing rate
        self.rate = rospy.Rate(30)  # 30 Hz
        
    def convert_to_pointcloud(self, depth_image):
        # Convert depth image to point cloud
        height, width = depth_image.shape
        points = []
        
        for i in range(height):
            for j in range(width):
                z = depth_image[i,j]
                if z > 0:  # Filter out invalid depth values
                    x = (j - width/2) * z / width
                    y = (i - height/2) * z / height
                    points.append([x, y, z])
                    
        return np.array(points)
        
    def publish_pointcloud(self, points):
        # Create header
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_1_link"
        
        # Create fields
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Create and publish point cloud message
        pc_msg = pc2.create_cloud(header, fields, points)
        self.pub.publish(pc_msg)
        
    def run(self):
        while not rospy.is_shutdown():
            # Get depth image from vision sensor
            depth_image = self.vision_sensor.capture_depth()
            
            # Convert to point cloud
            points = self.convert_to_pointcloud(depth_image)
            
            # Publish
            self.publish_pointcloud(points)
            
            # Step simulation
            self.pr.step()
            
            self.rate.sleep()
            
    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()
        rospy.loginfo("Shutting down SimStreamer node")

if __name__ == '__main__':
    try:
        node = SimStreamer()
        node.run()
    except rospy.ROSInterruptException:
        node.shutdown()
