#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Vector3
from percept_interfaces.srv import AgentStateToCircForce
import time
import numpy as np

class ServiceTester(Node):
    def __init__(self):
        super().__init__('service_tester')
        
        # Create timer that calls service every second
        self.timer = self.create_timer(0.100, self.timer_callback)
        
        # Create service client
        self.client = self.create_client(AgentStateToCircForce, '/get_heuristic_circforce')
        
        # Wait for service to become available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        
        # Create publisher for markers
        self.marker_publisher = self.create_publisher(Marker, 'circ_force_markers', 10)

        self.start_time = time.time()
        self.start_pos = np.array([-2.0, -0.50, 1.0])  # Start 2m to the left and 0.5m offset
        self.velocity = np.array([0.1, 0.0, 0.0])    # Moving right at 1 m/s
        # self.velocity = self.velocity / np.linalg.norm(self.velocity)  # Normalize



    def timer_callback(self):
        # Create request with a single pose
        request = AgentStateToCircForce.Request()
        
        # Calculate current position based on elapsed time
        elapsed_time = time.time() - self.start_time
        current_pos = self.start_pos + self.velocity * elapsed_time
        
        # Reset position if we've gone too far
        if current_pos[0] > 2.0:  # Reset after moving 2m to the right
            self.start_time = time.time()
            current_pos = self.start_pos

        # Set agent position
        request.agent_pose = Pose()
        request.agent_pose.position.x = float(current_pos[0])
        request.agent_pose.position.y = float(current_pos[1])
        request.agent_pose.position.z = float(current_pos[2])
        request.agent_pose.orientation.w = 1.0

        # Set constant velocity
        request.agent_velocity = Vector3()
        request.agent_velocity.x = float(self.velocity[0])
        request.agent_velocity.y = float(self.velocity[1])
        request.agent_velocity.z = float(self.velocity[2])

        # Set goal pose at origin
        request.target_pose = Pose()
        request.target_pose.position.x = 0.0
        request.target_pose.position.y = 0.0
        request.target_pose.position.z = 1.0
        request.target_pose.orientation.w = 1.0
    
        # Set detection shell radius parameter
        request.detect_shell_rad = 2.0

        # Send request and time the response
        start_time = time.time()
        future = self.client.call_async(request)
        
        # data sent as request
        self.get_logger().info('Request:\n' + 
                             f'  agent_pos = [{request.agent_pose.position.x:.3f}, {request.agent_pose.position.y:.3f}, {request.agent_pose.position.z:.3f}]\n' +
                             f'  agent_vel = [{request.agent_velocity.x:.3f}, {request.agent_velocity.y:.3f}, {request.agent_velocity.z:.3f}]\n' +
                             f'  target_pos  = [{request.target_pose.position.x:.3f}, {request.target_pose.position.y:.3f}, {request.target_pose.position.z:.3f}]\n' +
                             f'  detect_rad = {request.detect_shell_rad:.3f}')
        # Add callback for when response is received
        future.add_done_callback(
            lambda future: self.response_callback(future, start_time, request))

    def response_callback(self, future, start_time, request):
        try:
            response = future.result()
            elapsed_time = time.time() - start_time
            
            self.get_logger().info('Response:\n' +
                                 f'  Time elapsed: {elapsed_time:.5f} seconds\n' +
                                 f'  Circ force:   [{response.circ_force.x:.3f}, {response.circ_force.y:.3f}, {response.circ_force.z:.3f}]\n' #+f'  Within radius: {response.not_null}\n'
                                 )
            
            # Publish markers for agent position and circular force
            self.publish_markers(response, request)
            
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def publish_markers(self, response, request):
        # Agent position as a sphere
        agent_marker = Marker()
        agent_marker.header.frame_id = "map"
        agent_marker.header.stamp = self.get_clock().now().to_msg()
        agent_marker.ns = "agent"
        agent_marker.id = 0
        agent_marker.type = Marker.SPHERE
        agent_marker.action = Marker.ADD
        agent_marker.pose = request.agent_pose
        agent_marker.scale.x = 0.2  # Sphere size
        agent_marker.scale.y = 0.2
        agent_marker.scale.z = 0.2
        agent_marker.color.r = 0.0
        agent_marker.color.g = 1.0
        agent_marker.color.b = 0.0
        agent_marker.color.a = 1.0

        # Circular force vector as an arrow
        force_marker = Marker()
        force_marker.header.frame_id = "map"
        force_marker.header.stamp = self.get_clock().now().to_msg()
        force_marker.ns = "circ_force"
        force_marker.id = 1
        force_marker.type = Marker.ARROW
        force_marker.action = Marker.ADD
        force_marker.scale.x = 0.05  # Shaft diameter
        force_marker.scale.y = 0.1   # Arrowhead diameter
        force_marker.scale.z = 0.1
        force_marker.color.r = 1.0
        force_marker.color.g = 0.0
        force_marker.color.b = 0.0
        force_marker.color.a = 1.0

        # Define start and end points of the force arrow
        start_point = Point()
        start_point.x = request.agent_pose.position.x
        start_point.y = request.agent_pose.position.y
        start_point.z = request.agent_pose.position.z

        end_point = Point()
        end_point.x = request.agent_pose.position.x + response.circ_force.x
        end_point.y = request.agent_pose.position.y + response.circ_force.y
        end_point.z = request.agent_pose.position.z + response.circ_force.z

        force_marker.points.append(start_point)
        force_marker.points.append(end_point)

        # Goal position as a sphere
        goal_marker = Marker()
        goal_marker.header.frame_id = "map"
        goal_marker.header.stamp = self.get_clock().now().to_msg()
        goal_marker.ns = "goal"
        goal_marker.id = 2
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose = request.target_pose
        goal_marker.scale.x = 0.2  # Sphere size
        goal_marker.scale.y = 0.2
        goal_marker.scale.z = 0.2
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 1.0

        # Publish markers
        self.marker_publisher.publish(agent_marker)
        self.marker_publisher.publish(force_marker)
        self.marker_publisher.publish(goal_marker)

def main(args=None):
    rclpy.init(args=args)
    service_tester = ServiceTester()
    rclpy.spin(service_tester)
    service_tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
