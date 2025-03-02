#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import random

class SmoothRandomJointStatePublisher(Node):
    def __init__(self):
        super().__init__('smooth_random_joint_state_publisher')
        # Publisher for JointState messages
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        # Timer callback at 10 Hz
        self.timer = self.create_timer(0.030, self.timer_callback)
        # Joint names for the Panda robot:
        # 7 arm joints + 2 finger joints.
        self.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6',
            'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2'
        ]
        # Initialize joint positions (starting at zero for all joints)
        self.joint_states = [0.0 for _ in self.joint_names]
        # Standard deviation for the small random increment (adjust for smoother or more dynamic motion)
        self.increment_std_dev = 0.05

    def timer_callback(self):
        # Update each joint with a small random increment for smooth motion.
        for i in range(len(self.joint_states)):
            delta = random.gauss(0, self.increment_std_dev)
            self.joint_states[i] += delta

        # Create and fill the JointState message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_states

        # Publish the message
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing joint positions: {msg.position}')

def main(args=None):
    rclpy.init(args=args)
    node = SmoothRandomJointStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
