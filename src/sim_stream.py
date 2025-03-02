#!/usr/bin/env python3

import gymnasium as gym
import mani_skill
import torch

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import PointCloud2, PointField, Image
# import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros

from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig

from percept.utils.pose_helpers import create_tf_msg_from_xyzrpy
from percept.utils.troubleshoot import get_error_text

import trimesh
import numpy as np
import time

# import sapien
# from mani_skill.utils import sapien_utils, common
# from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from typing import Any, Dict, Union
from mani_skill.agents.robots import Fetch, Panda

@register_env("SimEnv-v1")
class SimEnv(PickCubeEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, 
                 robot_uids="panda", 
                 camera_configs=None, 
                 **kwargs):
        self.camera_configs = camera_configs if camera_configs is not None else []
        # Call parent class constructor after setting camera_configs
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        _camera_configs = []
        for config in self.camera_configs:
            _camera_configs.append(CameraConfig(
                config['name'],
                pose=config['pose'],
                width=config['width'],
                height=config['height'],             
                fov=config['fov'],
                near=config['near'],
                far=config['far']
            ))
        return _camera_configs

    # def _load_agent(self, options: dict):
    #     super()._load_agent(options)

class SimStreamer(Node):
    # Streams Scene from CoppeliaSim to ROS and back!
    def __init__(self):
        super().__init__('sim_streamer')

        self.logger = self.get_logger()

        # parameters
        self.declare_parameter('debug_mode', False)
        self.debug_mode = self.get_parameter('debug_mode').value

        self.declare_parameter('sim_config', "")
        self.sim_config_filepath = self.get_parameter('sim_config').value

        self.declare_parameter('env_id', "SimEnv-v1")
        self.env_id = self.get_parameter('env_id').value

        self.declare_parameter('disable_render', False)
        self.disable_render = self.get_parameter('disable_render').value

        self.declare_parameter('disable_publishing', False)
        self.disable_publishing = self.get_parameter('disable_publishing').value
        
        # setup
        self._setup()

    def _setup(self):

        if self.sim_config_filepath == "":
            self.logger.error("sim_config is not set")
            raise ValueError("sim_config is not set")

        self._load_config()
        self._setup_cameras()
        self._setup_sim()
        self._setup_publishers()
        self._setup_timer()


    def _load_config(self):
        import yaml

        with open(self.sim_config_filepath, 'r') as f:
            self.sim_config = yaml.safe_load(f)

        self.scene_config = self.sim_config['scene_config']
        self.static_camera_config = self.sim_config['static_camera_config']


    def _setup_cameras(self):
        # Create static transform broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # create camera configs
        self.cameras = {}
        static_transforms = []

        self.logger.info(f"static_camera_config: {self.static_camera_config}")
        for camera_name, camera_config in self.static_camera_config.items():
            # create static transform
            child_frame = f'{camera_name}_link'
            position = camera_config['pose']['position']
            orientation = camera_config['pose']['orientation']
            transform_msg = create_tf_msg_from_xyzrpy(
                child_frame,
                position["x"], position["y"], position["z"],  # position
                orientation["roll"], orientation["pitch"], orientation["yaw"]   # rotation
            )
            static_transforms.append(transform_msg)

            # create mani-skill camera config
            msk_camera_config = dict(
                name=camera_name,
                width=camera_config['resolution'][0],
                height=camera_config['resolution'][1],
                pose=Pose.create_from_pq(
                    p=[transform_msg.transform.translation.x, 
                        transform_msg.transform.translation.y, 
                        transform_msg.transform.translation.z], 
                    q=[transform_msg.transform.rotation.x, 
                        transform_msg.transform.rotation.y, 
                        transform_msg.transform.rotation.z,
                        transform_msg.transform.rotation.w
                    ]),
                fov=1.57,   # ~90° FOV
                near=0.01,
                far=10,                             
            )
            self.cameras[camera_name] = msk_camera_config
        # Broadcast all transforms
        self.tf_broadcaster.sendTransform(static_transforms)


    def _setup_sim(self):
        self.env = None
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        self.get_logger().info(f"Using device: {device}")
        self.env = gym.make(
            self.env_id, 
            obs_mode="depth", 
            # control_mode="pd_joint_pos", 
            render_mode="human" if not self.disable_render else 'sensors',
            camera_configs=list(self.cameras.values())
        )
        obs = self.env.reset()
        self.env.unwrapped.print_sim_details()


    def _setup_publishers(self):
        self.publishers_ = dict()

        # Set up depth image publishers for each camera
        for camera_name in self.cameras.keys():
            self.publishers_[f"{camera_name}_depth"] = self.create_publisher(
                Image,
                f'/cameras/{camera_name}/depth/image_raw',
                1
            )

            self.publishers_[camera_name] = self.create_publisher(
                PointCloud2, 
                f'/cameras/{camera_name}/depth/color/points', 
                1
            )

    def _setup_timer(self):
        self.timer_period = 1.0 if self.debug_mode else 0.025
        self.timer = self.create_timer(self.timer_period, self.run_sim)

    def run_sim(self):
        try:
            if rclpy.ok():
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                if not self.disable_publishing:
                    self._publish_pointclouds(obs)
                    self._publish_depth_images(obs)

        except KeyboardInterrupt:
            self.env.close()
            self.logger.info("Environment closed")
        except Exception as e:
            self.logger.error(get_error_text(e, print_stack_trace=True))


    def _publish_depth_images(self, obs):
        for camera_name in self.cameras.keys():
            depth_image = obs['sensor_data'][camera_name]['depth']
            depth_array = np.squeeze(depth_image.cpu().numpy())

            # Create Image message
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f'{camera_name}_link'
            msg.height = depth_array.shape[0]
            msg.width = depth_array.shape[1]
            msg.encoding = '32FC1'  # floating point depth values
            msg.is_bigendian = False
            msg.step = msg.width * 4  # 4 bytes per float32
            msg.data = depth_array.astype(np.float32).tobytes()

            self.publishers_[f"{camera_name}_depth"].publish(msg)

    def _publish_pointclouds(self, obs):
        # Get sensor data for each camera
        for camera_name, camera_config in self.cameras.items():
                # Get depth image from observation and remove extra dimensions
                depth_tensor = obs['sensor_data'][camera_name]['depth']
                depth_array = np.squeeze(depth_tensor.cpu().numpy())  # shape: (height, width)
                height, width = depth_array.shape

                # Compute focal length (assuming camera_config['fov'] is horizontal FOV in radians)
                fx = width / (2.0 * np.tan(camera_config['fov'] / 2.0))
                fy = height / (2.0 * np.tan(camera_config['fov'] / 2.0))

                # Create a meshgrid of pixel coordinates
                u = np.arange(width)
                v = np.arange(height)
                u_grid, v_grid = np.meshgrid(u, v)

                # Assume the principal point is at the image center
                cx = width / 2.0
                cy = height / 2.0

                # Convert pixel coordinates to 3D camera coordinates
                # Note: Adjusting coordinate system to match ROS convention
                # ROS: x forward, y left, z up
                # Converting from camera coordinates (z forward, x right, y down)
                z_3d = depth_array
                x_3d = -(u_grid - cx) * z_3d / fx  # Negative to flip x-axis
                y_3d = -(v_grid - cy) * z_3d / fy  # Negative to flip y-axis

                # Rearrange axes to match ROS convention (x forward, y left, z up)
                points = np.stack((z_3d, -x_3d, -y_3d), axis=-1)

                # Create and publish the point cloud message
                msg = self._create_point_cloud_msg(points, camera_name)
                self.publishers_[camera_name].publish(msg)

    def _create_point_cloud_msg(self, points_array, camera_name):
        # points_array is of shape (height, width, 3)
        height, width, _ = points_array.shape
        # Flatten the points row-wise
        points = points_array.reshape(-1, 3)

        frame_id = f'{camera_name}_link'
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        # Define the fields x, y, z.
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Now create an organized point cloud message
        msg.height = height
        msg.width = width
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12  # 3 * 4 bytes per point (float32)
        msg.row_step = msg.point_step * width
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()

        return msg


if __name__ == '__main__':
    rclpy.init()
    
    try:
        node = SimStreamer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.env.close()
        node.destroy_node()
        rclpy.shutdown()
