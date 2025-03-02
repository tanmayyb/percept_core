#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from percept.utils.pose_helpers import create_tf_msg_from_xyzrpy
from percept.utils.config_helpers import load_yaml_as_dict

from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor


def load_configs(node):
    static_camera_config = load_yaml_as_dict(node, 'static_camera_config')
    static_agent_config = load_yaml_as_dict(node, 'static_agent_config')
    return static_camera_config, static_agent_config

def load_static_tf_frames(node): # using config file
    static_camera_config, static_agent_config = load_configs(node)

    # create tf messages
    static_tf_frames = list()
    
    # camera frames
    camera_frames = list()
    for camera_name, camera_config in static_camera_config.items():
        position = camera_config['pose']['position']
        orientation = camera_config['pose']['orientation']
        child_frame = f'{camera_name}_link'
        msg = create_tf_msg_from_xyzrpy(child_frame, 
            position['x'], position['y'], position['z'],
            orientation['roll'], orientation['pitch'], orientation['yaw']
        )
        camera_frames.append(msg)

    # agent frame
    agent_frames=list()
    for agent_name, agent_config in static_agent_config.items():
        agent_frame = f'{agent_name}_frame'
        agent_pose = agent_config['pose']
        msg = create_tf_msg_from_xyzrpy(agent_frame, 
            agent_pose['position']['x'], agent_pose['position']['y'], agent_pose['position']['z'],
            agent_pose['orientation']['roll'], agent_pose['orientation']['pitch'], agent_pose['orientation']['yaw']
        )
    agent_frames.append(msg)

    # combine all frames
    static_tf_frames = camera_frames + agent_frames

    return static_tf_frames


def main():
    rclpy.init()
    node = Node('static_tf_publisher')

    # Declare parameters
    node.declare_parameter('static_camera_config', 
        Parameter.Type.STRING,
        ParameterDescriptor(description='Path to static camera configuration file'))
    node.declare_parameter('static_agent_config', 
        Parameter.Type.STRING,
        ParameterDescriptor(description='Path to static agent configuration file'))


    # Load transforms
    static_tf_frames = load_static_tf_frames(node)
    if not static_tf_frames:
        node.get_logger().error("No valid static transforms found. Exiting.")
        return

    # Publish static transforms
    broadcaster = tf2_ros.StaticTransformBroadcaster(node)
    broadcaster.sendTransform(static_tf_frames)

    node.get_logger().info("Static transforms published successfully.")

    return node


if __name__ == '__main__':
    try:
        node = main()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
