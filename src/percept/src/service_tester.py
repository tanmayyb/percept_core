#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from percept_interfaces.srv import PosesToVectors
import time
import numpy as np

class ServiceTester(Node):
    def __init__(self):
        super().__init__('service_tester')
        
        # Create timer that calls service every second
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Create service client
        self.client = self.create_client(PosesToVectors, '/get_heuristic_fields')
        
        # Wait for service to become available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def timer_callback(self):
        # Create request with sample poses
        request = PosesToVectors.Request()
        request.poses = []
        request.radius = 2.0  # Set radius parameter
        
        # Add some sample poses
        # Sample 1000 points in a 2m radius sphere around origin
        for i in range(1000):
            # Generate random spherical coordinates
            theta = 2 * 3.14159 * np.random.random() # azimuthal angle
            phi = np.arccos(2 * np.random.random() - 1) # polar angle
            r = 2.0 * np.random.random() # radius up to 2m
            
            # Convert to cartesian coordinates
            pose = Pose()
            pose.position.x = r * np.sin(phi) * np.cos(theta)
            pose.position.y = r * np.sin(phi) * np.sin(theta) 
            pose.position.z = r * np.cos(phi)
            pose.orientation.w = 1.0
            request.poses.append(pose)
        
        # Send request and time the response
        start_time = time.time()
        future = self.client.call_async(request)
        
        # Add callback for when response is received
        future.add_done_callback(
            lambda future: self.response_callback(future, start_time))

    def response_callback(self, future, start_time):
        try:
            response = future.result()
            elapsed_time = time.time() - start_time
            
            self.get_logger().info(
                f'Service call completed in {elapsed_time:.3f} seconds')
            self.get_logger().info(
                f'Received {len(response.vectors)} vectors')
            self.get_logger().info(
                f'e.g. {response.vectors[:100]}')


            # self.get_logger().info(
            #     f'Within radius status: {response.within_radius}')
            
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    service_tester = ServiceTester()
    rclpy.spin(service_tester)
    service_tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
