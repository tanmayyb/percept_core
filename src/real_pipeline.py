#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import argparse
import percept.utils.troubleshoot as troubleshoot

from percept.perception_pipeline import PerceptionPipeline
from percept.perception_node import PerceptionNode
from percept.utils.pose_helpers import create_tf_matrix_from_euler
# from percept.utils.config_helpers import load_yaml_as_dict

from sensor_msgs.msg import PointCloud2, JointState

from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

from ament_index_python.packages import get_package_share_directory
from pathlib import Path

import yaml

class RealPerceptionPipeline(PerceptionPipeline):
    def __init__(self, 
                 node,
                 configs,
                 enable_dynamic_cameras,
                 enable_dynamic_agents,
                 enable_robot_body_subtraction, 
                 show_pipeline_delays, 
                 show_total_pipeline_delay
        ):
        super().__init__(node)
        self.enable_dynamic_cameras = enable_dynamic_cameras
        self.enable_dynamic_agents = enable_dynamic_agents
        self.enable_robot_body_subtraction = enable_robot_body_subtraction
        self.show_pipeline_delays = show_pipeline_delays
        self.show_total_pipeline_delay = show_total_pipeline_delay

        # load configs
        self.configs = configs
        self._load_and_setup_pipeline_configs()

        # finish setup
        super().setup()

    def _load_and_setup_pipeline_configs(self):        
        pipeline_config = self.configs['pipeline_config']['perception_pipeline_config']
        camera_configs = self.configs['camera_configs']
        agent_configs = self.configs['agent_configs']
        robot_urdf_filepath = self.configs['robot_urdf_filepath']
        
        self.scene_bounds = pipeline_config['scene_bounds']
        self.voxel_resolution = pipeline_config['voxel_props']['voxel_resolution']
        
        if self.enable_robot_body_subtraction:
            self.load_robot_urdf(robot_urdf_filepath)
        
        self._load_camera_configs(camera_configs)
        self._load_agent_configs(agent_configs)

    def _load_camera_configs(self, camera_configs):
        if self.enable_dynamic_cameras:
            self.camera_names = list(camera_configs.keys())
        else:
            self.camera_names = list(camera_configs.keys())
            self.camera_tfs = dict()
            for camera_name in self.camera_names:
                self.camera_tfs[camera_name] = create_tf_matrix_from_euler(camera_configs[camera_name]['pose'])
                self.node.get_logger().info(f"static camera '{camera_name}' setup complete")

    def _load_agent_configs(self, agent_configs):
        if self.enable_dynamic_agents:
            self.agent_names = list(agent_configs.keys())
        else:
            self.agent_names = list(agent_configs.keys())
            self.agent_tfs = dict()
            for agent_name in self.agent_names:
                self.agent_tfs[agent_name] = create_tf_matrix_from_euler(agent_configs[agent_name]['pose'])

class RealPerceptionNode(PerceptionNode):
    def __init__(self):
        super().__init__('real_perception_node')

        # declare parameters
        self.declare_parameter(
            'enable_dynamic_cameras', False, ParameterDescriptor(description='Use static scene configuration instead of listening to a topic'))
        self.declare_parameter(
            'enable_dynamic_agents', False, ParameterDescriptor(description='Use static agent configuration instead of listening to a topic'))
        self.declare_parameter(
            'enable_robot_body_subtraction', False, ParameterDescriptor(description='Enable robot body subtraction'))
        self.declare_parameter(
            'show_pipeline_delays', False, ParameterDescriptor(description='Show pipeline delays'))
        self.declare_parameter(
            'show_total_pipeline_delay', False, ParameterDescriptor(description='Show total pipeline delay'))

        self.enable_dynamic_cameras = self.get_parameter('enable_dynamic_cameras').get_parameter_value().bool_value
        self.enable_dynamic_agents = self.get_parameter('enable_dynamic_agents').get_parameter_value().bool_value
        self.enable_robot_body_subtraction = self.get_parameter('enable_robot_body_subtraction').get_parameter_value().bool_value
        show_pipeline_delays = self.get_parameter('show_pipeline_delays').get_parameter_value().bool_value
        show_total_pipeline_delay = self.get_parameter('show_total_pipeline_delay').get_parameter_value().bool_value

        pipeline_configs = self._create_pipeline_configs()

        # Initialize pipeline
        self.pipeline = RealPerceptionPipeline(
            self,
            pipeline_configs,
            self.enable_dynamic_cameras,
            self.enable_dynamic_agents,
            self.enable_robot_body_subtraction,
            show_pipeline_delays,
            show_total_pipeline_delay
        )

        # setup subscribers
        self.setup_ros_subscribers()

    def _create_pipeline_configs(self)->dict:
        enable_dynamic_cameras = self.enable_dynamic_cameras
        enable_dynamic_agents = self.enable_dynamic_agents
        enable_robot_body_subtraction = self.enable_robot_body_subtraction

        def load(filepath):
            with open(filepath, "r") as f:
                return yaml.load(f, Loader=yaml.SafeLoader)

        configs = {}

        package_share_dir = Path(get_package_share_directory('percept'))

        if enable_dynamic_cameras:
            camera_configs_filepath = package_share_dir / 'config' / 'dynamic_camera_configs.yaml'
        else:
            camera_configs_filepath = package_share_dir / 'config' / 'static_cameras_setup.yaml'

        if enable_dynamic_agents:
            agent_configs_filepath = package_share_dir / 'config' / 'dynamic_agent_configs.yaml'
        else:
            agent_configs_filepath = package_share_dir / 'config' / 'static_agents_setup.yaml'

        if enable_robot_body_subtraction:
            robot_urdf_filepath = package_share_dir / 'assets'/ 'robots' / 'panda.urdf'
        else:
            robot_urdf_filepath = None

        pipeline_config_filepath = package_share_dir / 'config' / 'perception_pipeline_setup.yaml'

        configs['camera_configs'] = load(camera_configs_filepath)
        configs['agent_configs'] = load(agent_configs_filepath)
        configs['pipeline_config'] = load(pipeline_config_filepath)
        configs['robot_urdf_filepath'] = robot_urdf_filepath
        return configs


    def setup_ros_subscribers(self):
        self.subscribers = {}
        if self.pipeline.camera_names:
            for camera_name in self.pipeline.camera_names:
                topic = f'/cameras/{camera_name}/depth/color/points'
                self.subscribers[camera_name] = self.create_subscription(
                    PointCloud2, 
                    topic,
                    lambda msg, cn=camera_name: self.static_camera_callback(msg, cn),
                    10)
                self.get_logger().info(f"subscribed to {camera_name} topic")
        # if self.pipeline.agent_names:
        #     for agent_name in self.pipeline.agent_names:
        #         topic = f'/agents/{agent_name}/pose'
        #         self.subscribers[agent_name] = self.create_subscription(
        #             Pose,
        #             topic,
        #             lambda msg, cn=agent_name: self.static_agent_callback(msg, cn),
        #             10)

    def static_camera_callback(self, msg, camera_name):
        with self.buffer_lock:
            self.pointcloud_buffer[camera_name] = msg
            if len(self.pointcloud_buffer) == len(self.pipeline.camera_names):
                buffer_copy = self.pointcloud_buffer.copy()
                if not self.enable_dynamic_cameras:
                    camera_tfs = self.pipeline.camera_tfs
                if not self.enable_dynamic_agents:
                    agent_tfs = self.pipeline.agent_tfs
                if not self.enable_robot_body_subtraction:
                    joint_state = None
                else: 
                    joint_state = None
                future = self.thread_pool.submit(self.run_pipeline, buffer_copy, camera_tfs, agent_tfs, joint_state)
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
