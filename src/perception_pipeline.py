#!/usr/bin/env python3
import time
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cupoch as cph
import numpy as np
import cupy as cp
import numba.cuda as cuda
import percept.utils.troubleshoot as troubleshoot
from percept.utils.pose_helpers import create_tf_matrix_from_euler
from pathlib import Path

from utils.temporal_filter_ema import TemporalEMAVoxelFilter
from utils.temporal_filter_majority import TemporalMajorityVoxelFilter
from utils.temporal_filter_median import TemporalMedianVoxelFilter

class PerceptionPipeline():
    def __init__(self, node):
        self.show_pipeline_delays = False
        self.show_total_pipeline_delay = False
        self.enable_robot_body_subtraction = False

        # filtering params
        self.enable_spatial_filtering = True
        self.enable_temporal_filtering = True

        # downsampling params
        self.downsample = 5

        # spatial filtering params
        self.outlier_neighbours = 2
        self.outlier_std_ratio = 2.0

        # temporal filter params
        self.temporal_filter_type = 'median'

        self.ema_alpha = 0.70            # Responsiveness to new observations
        self.ema_decay_factor = 0.85    # How long absent voxels persist
        self.ema_confidence_threshold = 0.70   # Minimum confidence for output            

        self.majority_window_size = 5 # number of frames to consider
        
        self.median_window_size = 5 # number of frames to consider
        self.median_threshold = 0.85 # threshold for median filtering

        # robot filtering params
        self.robot_aabb_scale = 1.25

        self.node = node
        self.logger = node.get_logger().get_child('perception_pipeline')
        self.check_cuda()
        # Create process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        # Create thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def check_cuda(self):
        """Check if CUDA is available using nvidia-smi"""
        try:
            output = subprocess.check_output(["nvidia-smi"])
            self.logger.info("CUDA is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("CUDA is not available - nvidia-smi command failed")
            raise RuntimeError("CUDA is required for this pipeline")

    def load_robot_description(self, package_share_dir:Path, robot_id:str):
        self.robot_id = robot_id
        if robot_id == 'panda':
            robot_description_path = package_share_dir / 'assets' / 'robots' / 'franka_panda' / 'panda.urdf'
        else:
            raise ValueError(f"Robot ID {robot_id} not supported")
        
        try:
            self.robot_kinematics_chain = cph.kinematics.KinematicChain(str(robot_description_path))
        except Exception as e:
            self.logger.error(troubleshoot.get_error_text(e))
            raise RuntimeError("Failed to load robot URDF")


    def setup(self):
        # set scene geometry properties
        min_bound = np.array(self.scene_bounds['min'])
        max_bound = np.array(self.scene_bounds['max'])

        self.voxel_min_bound = tuple(min_bound)
        self.voxel_max_bound = tuple(max_bound)
        
        if not np.all(np.abs(max_bound - min_bound) - (max_bound[0] - min_bound[0]) < 1e-6):
            raise ValueError("The min_bound and max_bound do not define a cube. Make it a cube!")

        self.scene_bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        self.max_voxels = np.prod((max_bound-min_bound)/self.voxel_size).astype(int)

        if self.temporal_filter_type == 'ema':
            self.temporal_voxel_filter = TemporalEMAVoxelFilter(
                alpha=self.ema_alpha, 
                decay_factor=self.ema_decay_factor,   
                confidence_threshold=self.ema_confidence_threshold,
                max_voxels=self.max_voxels
            )
        elif self.temporal_filter_type == 'majority':
            self.temporal_voxel_filter = TemporalMajorityVoxelFilter(
                window_size=self.majority_window_size,
                logger=self.logger
            )
        elif self.temporal_filter_type == 'median':
            self.temporal_voxel_filter = TemporalMedianVoxelFilter(
                window_size=self.median_window_size,
                threshold=self.median_threshold,
                max_voxels=self.max_voxels,
                logger=self.logger
            )

    def _process_single_pointcloud(self, camera_name:str, msg, camera_tf=None):
        downsample = self.downsample
        scene_bbox = self.scene_bbox
        try:
            # Create CUDA stream for parallel GPU operations
            stream = cp.cuda.Stream()
            with stream:
                # load pointcloud from ROS msg
                pcd = cph.geometry.PointCloud()
                msg_bytes = bytes(msg.data)
                temp = cph.io.create_from_pointcloud2_msg(
                    msg_bytes, cph.io.PointCloud2MsgInfo.default_dense(
                        msg.width, msg.height, msg.point_step)
                )       
                pcd.points = temp.points
                
                # downsample increases performance
                if downsample>0:
                    pcd = pcd.uniform_down_sample(downsample)
                
                # Transform after downsampling to reduce computation
                if camera_tf is not None:
                    pcd = pcd.transform(camera_tf)

                # Crop pointcloud according to scene bounds
                pcd = pcd.crop(scene_bbox)

            return pcd
        except Exception as e:
            self.logger.error(troubleshoot.get_error_text(e))

    def _parse_pointclouds(self, pointclouds:dict, camera_tfs:dict):
        start = time.time()

        # Early return if pointclouds is None
        if pointclouds is None:
            self.logger.error("Received None pointclouds")
            return None

        # Process point clouds in parallel using threads with batching
        batch_size = 3  # Process 3 cameras at once
        processed_pointclouds = {}
        
        for i in range(0, len(self.camera_names), batch_size):
            batch_cameras = self.camera_names[i:i + batch_size]
            futures = []
            
            for camera_name in batch_cameras:
                futures.append(
                    self.thread_pool.submit(
                        self._process_single_pointcloud,
                        camera_name,
                        pointclouds[camera_name],
                        camera_tfs.get(camera_name)
                    )
                )
            
            # Wait for batch to complete before starting next batch
            for camera_name, future in zip(batch_cameras, futures):
                try:
                    processed_pointclouds[camera_name] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing {camera_name}: {troubleshoot.get_error_text(e)}")

        if self.show_pipeline_delays:
            self.logger.info(f"PCD Processing (CPU+GPU) [sec]: {time.time()-start}")

        return processed_pointclouds
    
    def _merge_pointclouds(self, pointclouds:dict):
        # merge pointclouds
        start = time.time()

        def direct_merge(
            source_gpu:cph.geometry.PointCloud,
            target_gpu:cph.geometry.PointCloud
        ):
            return source_gpu+target_gpu

        source_gpu = pointclouds[self.camera_names[0]]
        merged_gpu = source_gpu
        if len(self.camera_names)>1:
            target_gpu = pointclouds[self.camera_names[1]]
            merged_gpu = direct_merge(merged_gpu, target_gpu)
        if len(self.camera_names)>2:
            for camera_name in self.camera_names[2:]:
                target_gpu = pointclouds[camera_name]
                merged_gpu = direct_merge(merged_gpu, target_gpu)

        if self.show_pipeline_delays:
            self.logger.info(f"Registration (GPU) [sec]: {time.time()-start}")
        return merged_gpu
    
    def _perform_robot_body_subtraction(self, pointcloud:dict, agent_tfs:dict, joint_states:dict):
        start = time.time()
        filtered_points = pointcloud
        for agent_name, agent_tf in agent_tfs.items():
            try:
                joint_mappings = {f'{self.robot_id}_{joint_name}': \
                        joint_states[f'{agent_name}_{joint_name}'] for joint_name in self.joint_names}
                
                poses = self.robot_kinematics_chain.forward_kinematics(joint_mappings, agent_tf)
                body_meshes = self.robot_kinematics_chain.get_transformed_visual_geometry_map(poses)

                for mesh in body_meshes.values():
                    aabb = mesh.get_axis_aligned_bounding_box().scale(self.robot_aabb_scale, center=True)
                    indices = aabb.get_point_indices_within_bounding_box(filtered_points.points)
                    filtered_points = filtered_points.select_by_index(indices, invert=True)
            except Exception as e:
                self.logger.error(f"Failed to get meshes for agent {agent_name}: {troubleshoot.get_error_text(e, print_stack_trace=False)}")
                continue

        if self.show_pipeline_delays:
            self.logger.info(f"Robot Filtering (GPU) [sec]: {time.time()-start}")
        return filtered_points

    def _perform_voxelization(self, pcd:cph.geometry.PointCloud):
        start = time.time()
        voxel_grid = cph.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=self.voxel_size,
            min_bound=self.voxel_min_bound,
            max_bound=self.voxel_max_bound,
        )
        if self.show_pipeline_delays:
            self.logger.info(f"Voxelization (GPU) [sec]: {time.time()-start}")

        voxel_keys = list(voxel_grid.voxels.cpu().keys())
        return voxel_grid, voxel_keys

    def _perform_temporal_filtering(self, voxel_keys):
        if self.enable_temporal_filtering:
            start = time.time()

            voxel_keys = self.temporal_voxel_filter.update(voxel_keys)

            if self.show_pipeline_delays:
                self.logger.info(f"Temporal Filtering (GPU) [sec]: {time.time()-start}")

        return voxel_keys

    def _perform_grid2world_transform(self, voxel_grid:cph.geometry.VoxelGrid, voxel_keys:list, filtered_voxel_keys:list):
        start = time.time()

        if len(voxel_keys) == 0:
            return None

        grid2pcd = cph.geometry.PointCloud(cph.utility.Vector3fVector(filtered_voxel_keys))
        grid2pcd = grid2pcd.translate(-cp.asnumpy(cp.min(cp.asarray(voxel_keys), axis=0)), relative=True)
        grid2pcd = grid2pcd.scale(self.voxel_size, center=False)
        grid2pcd = grid2pcd.translate(voxel_grid.get_min_bound()+self.voxel_size/2, relative=True)
        
        if self.show_pipeline_delays:
            self.logger.info(f"Grid2WorldTransform (CPU+GPU) [sec]: {time.time()-start}")
        return grid2pcd

    def _remove_outliers(self, pcd:cph.geometry.PointCloud):
        if self.enable_spatial_filtering:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.outlier_neighbours, 
                std_ratio=self.outlier_std_ratio
            )
        return pcd

    def run_pipeline(self, pointclouds:dict, camera_tfs:dict, agent_tfs:dict, joint_states:dict):
        # streamer/realsense gives pointclouds and tfs
        outlier_neighbours = self.outlier_neighbours
        enable_robot_body_subtraction = self.enable_robot_body_subtraction
        show_pipeline_delays = self.show_pipeline_delays
        show_total_pipeline_delay = self.show_total_pipeline_delay

        start = time.time()

        try:
            tmp = self._parse_pointclouds(pointclouds, camera_tfs)
            pointcloud = self._merge_pointclouds(tmp)
        except Exception as e:
            self.logger.error(troubleshoot.get_error_text(e))
            return None, None
        try:
            if enable_robot_body_subtraction:
                pointcloud = self._perform_robot_body_subtraction(pointcloud, agent_tfs, joint_states)

            if pointcloud is not None:
                voxel_grid, voxel_keys = self._perform_voxelization(pointcloud)

                filtered_voxel_keys = self._perform_temporal_filtering(voxel_keys)
                g2w_pcd = self._perform_grid2world_transform(voxel_grid, voxel_keys, filtered_voxel_keys)
                output_pcd = self._remove_outliers(g2w_pcd)
            else:
                output_pcd = None

        except Exception as e:
            self.logger.error(troubleshoot.get_error_text(e), print_stack_trace=True)
            return None

        if show_total_pipeline_delay:
            self.logger.info(f"Perception Pipeline (CPU+GPU) [sec]: {time.time()-start}")
        return np.array(output_pcd.points.cpu())