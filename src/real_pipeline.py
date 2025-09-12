#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import argparse
import percept.utils.troubleshoot as troubleshoot

from percept.perception_pipeline import PerceptionPipeline
from percept.perception_node import PerceptionNode
from percept.utils.pose_helpers import create_tf_matrix_from_euler
# from percept.utils.config_helpers import load_yaml_as_dict
import cupoch as cph

from sensor_msgs.msg import PointCloud2, JointState

from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from ament_index_python.packages import get_package_share_directory

from pathlib import Path
from copy import deepcopy
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
        package_share_dir = self.configs['package_share_dir']
        pipeline_config = self.configs['pipeline_config']['perception_pipeline_config']
        camera_configs = self.configs['camera_configs']
        agent_configs = self.configs['agent_configs']
        robot_id = self.configs['robot_id']
        
        self.scene_bounds = pipeline_config['scene_bounds']
        self.voxel_size = pipeline_config['graphics_settings']['voxel_size']
        self.exclusion_region_bbs =  [ cph.geometry.AxisAlignedBoundingBox(exclusion_region_bb[0], exclusion_region_bb[1]) for exclusion_region_bb in pipeline_config['exclusion_region_bbs']]
        
        if self.enable_robot_body_subtraction:
            self.load_robot_description(package_share_dir, robot_id)
        
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
        agents = agent_configs['agents']
        configurations = agent_configs['configurations']
        if self.enable_dynamic_agents:
            self.agent_names = list(agents.keys())
        else:
            self.agent_names = list(agents.keys())
            self.agent_tfs = dict()
            for agent_name in self.agent_names:
                self.agent_tfs[agent_name] = create_tf_matrix_from_euler(agents[agent_name]['pose'])
        self.joint_names = configurations['joints']

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

        # Log all parameters
        self.get_logger().info("Real Perception Pipeline Parameters:")
        self.get_logger().info(f"\t enable_dynamic_cameras: {self.enable_dynamic_cameras}")
        self.get_logger().info(f"\t enable_dynamic_agents: {self.enable_dynamic_agents}")
        self.get_logger().info(f"\t enable_robot_body_subtraction: {self.enable_robot_body_subtraction}")
        self.get_logger().info(f"\t show_pipeline_delays: {show_pipeline_delays}")
        self.get_logger().info(f"\t show_total_pipeline_delay: {show_total_pipeline_delay}")

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
            robot_id = 'panda'
        else:
            robot_id = None

        pipeline_config_filepath = package_share_dir / 'config' / 'perception_pipeline_setup.yaml'

        return {
            'package_share_dir': package_share_dir,
            'camera_configs': load(camera_configs_filepath),
            'agent_configs': load(agent_configs_filepath),
            'pipeline_config': load(pipeline_config_filepath),
            'robot_id': robot_id
        }


    def setup_ros_subscribers(self):
        self.subscribers = {}
        if self.pipeline.camera_names:
            for camera_name in self.pipeline.camera_names:
                topic = f'/cameras/{camera_name}/depth/color/points'
                self.subscribers[camera_name] = self.create_subscription(
                    PointCloud2, 
                    topic,
                    lambda msg, cn=camera_name: self._camera_callback(msg, cn),
                    10)
                self.get_logger().info(f"subscribed to {camera_name} topic")

        if self.enable_robot_body_subtraction:
            if self.pipeline.agent_names:
                self.subscribers['joint_states'] = self.create_subscription(
                    JointState,
                    '/joint_states',
                    lambda msg: self._joint_state_callback(msg),
                    10
                )
            self.num_joint_states = len(self.pipeline.agent_names)*len(self.pipeline.joint_names)

    def _joint_state_callback(self, msg):
        with self.joint_state_buffer_lock:
            for name, position in zip(msg.name, msg.position):
                self.joint_state_buffer[name] = position
            # self.get_logger().info(f"joint_state_buffer: {self.joint_state_buffer}")
            self._try_run_pipeline()

    def _camera_callback(self, msg, camera_name):
        with self.pointcloud_buffer_lock:
            self.pointcloud_buffer[camera_name] = msg
            self._try_run_pipeline()
            
    def _try_run_pipeline(self):
        # Check if we have all necessary data to run the pipeline
        all_cameras_ready = len(self.pointcloud_buffer) == len(self.pipeline.camera_names)
        joint_state_ready = len(self.joint_state_buffer) == self.num_joint_states
        
        if all_cameras_ready and joint_state_ready:
            pointcloud_buffer = self.pointcloud_buffer.copy()
            joint_state_buffer = deepcopy(self.joint_state_buffer)

            if not self.enable_dynamic_cameras:
                camera_tfs = self.pipeline.camera_tfs
            else:
                camera_tfs = None  # Dynamic camera case would need to obtain TFs differently
                
            if not self.enable_dynamic_agents:
                agent_tfs = self.pipeline.agent_tfs
            else:
                agent_tfs = None  # Dynamic agent case would need to obtain TFs differently
                
            # Run the pipeline with the gathered data
            future = self.thread_pool.submit(
                self.run_pipeline, 
                pointcloud_buffer, 
                camera_tfs, 
                agent_tfs, 
                joint_state_buffer
            )
            
            # Clear buffers after launching the pipeline
            self.pointcloud_buffer.clear()
            self.joint_state_buffer.clear()

def main():
    rclpy.init()
    node = RealPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down...")
    except Exception as e:
        node.get_logger().error(troubleshoot.get_error_text(e, print_stack_trace=True))

if __name__ == "__main__":
    main()
