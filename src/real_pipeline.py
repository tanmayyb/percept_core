#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import argparse
import percept.utils.troubleshoot as troubleshoot

from percept.perception_pipeline import PerceptionPipeline
from percept.perception_node import PerceptionNode
from percept.utils.camera_helpers import create_tf_matrix_from_euler
from percept.utils.config_helpers import load_yaml_as_dict

from sensor_msgs.msg import PointCloud2

from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor



class RealPerceptionPipeline(PerceptionPipeline):
    def __init__(self, node, load_static_scene_config=False, show_pipeline_delays=False, show_total_pipeline_delay=False):
        super().__init__(node)
        self.show_pipeline_delays = show_pipeline_delays
        self.show_total_pipeline_delay = show_total_pipeline_delay

        # load configs
        self.load_and_setup_pipeline_configs()

        if load_static_scene_config:
            self.node.get_logger().info("loading static camera configs...")
            self.load_and_setup_static_camera_configs()
        # else: TODO: setup for dynamic cameras

        # finish setup
        super().setup()

    def load_and_setup_pipeline_configs(self):        
        pipeline_config_data = load_yaml_as_dict(self.node, 'real_pipeline_config')
        self.perception_pipeline_config = pipeline_config_data['perception_pipeline_config']
        self.scene_bounds = self.perception_pipeline_config['scene_bounds']
        self.cubic_size = self.perception_pipeline_config['voxel_props']['cubic_size']
        self.voxel_resolution = self.perception_pipeline_config['voxel_props']['voxel_resolution']

    def load_and_setup_static_camera_configs(self):
        self.static_camera_config = load_yaml_as_dict(self.node, 'static_camera_config')
        # self.node.get_logger().info(f"static camera config: {self.static_camera_config}")
        def setup_static_cameras(static_camera_config):
            self.camera_names = list(static_camera_config.keys())
            self.tfs = dict()
            for camera_name in self.camera_names:
                self.tfs[camera_name] = create_tf_matrix_from_euler(static_camera_config[camera_name]['pose'])
                self.node.get_logger().info(f"static camera '{camera_name}' setup complete")
        
        setup_static_cameras(self.static_camera_config)


class RealPerceptionNode(PerceptionNode):
    def __init__(self):
        super().__init__('real_perception_node')

        # declare parameters
        self.declare_parameter('static_scene', 
            True,
            ParameterDescriptor(description='Use static scene configuration instead of listening to a topic'))
        self.declare_parameter('static_agent', 
            True,
            ParameterDescriptor(description='Use static agent configuration instead of listening to a topic'))
        self.declare_parameter('agent_config', 
            Parameter.Type.STRING,
            ParameterDescriptor(description='Path to agent position configuration file'))
        self.declare_parameter('real_pipeline_config', 
            Parameter.Type.STRING,
            ParameterDescriptor(description='Path to real pipeline configuration file'))
        self.declare_parameter('static_camera_config', 
            Parameter.Type.STRING,
            ParameterDescriptor(description='Path to camera configuration file'))
        self.declare_parameter('show_pipeline_delays', 
            False,
            ParameterDescriptor(description='Show pipeline delays'))
        show_pipeline_delays = self.get_parameter('show_pipeline_delays').get_parameter_value().bool_value
        self.declare_parameter('show_total_pipeline_delay', 
            False,
            ParameterDescriptor(description='Show total pipeline delay'))
        show_total_pipeline_delay = self.get_parameter('show_total_pipeline_delay').get_parameter_value().bool_value

        # Initialize pipeline
        self.pipeline = RealPerceptionPipeline(
            self,
            load_static_scene_config=self.get_parameter('static_scene').get_parameter_value().bool_value,
            show_pipeline_delays=show_pipeline_delays,
            show_total_pipeline_delay=show_total_pipeline_delay
        )

        # load static agent config if specified
        if  self.get_parameter('static_agent').get_parameter_value().bool_value:
            self.get_logger().info("loading static agent config...")
            self.load_and_setup_static_agent_config('agent_config')

        # setup subscribers
        if self.pipeline.camera_names:
            self.setup_ros_subscribers()


    def setup_ros_subscribers(self):
        self.subscribers = {}
        for camera_name in self.pipeline.camera_names:
            topic = f'/cameras/{camera_name}/depth/color/points'
            self.subscribers[camera_name] = self.create_subscription(
                PointCloud2, 
                topic,
                lambda msg, cn=camera_name: self.static_camera_callback(msg, cn),
                10)
            self.get_logger().info(f"subscribed to {camera_name} topic")

    def static_camera_callback(self, msg, camera_name):
        with self.buffer_lock:
            self.pointcloud_buffer[camera_name] = msg

            # TODO: move lock here
            # Check if we have data from all cameras
            if len(self.pointcloud_buffer) == len(self.pipeline.camera_names):
                buffer_copy = self.pointcloud_buffer.copy()
                tfs = self.pipeline.tfs # use static camera tfs
                future = self.thread_pool.submit(self.run_pipeline, buffer_copy, tfs, self.agent_pos, False)
                self.pointcloud_buffer.clear()

def main():
    rclpy.init()
    node = RealPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    except Exception as e:
        node.get_logger().error(troubleshoot.get_error_text(e))

if __name__ == "__main__":
    main()
