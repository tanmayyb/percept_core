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

class PerceptionPipeline():
    def __init__(self, node):
        self.show_pipeline_delays = False
        self.show_total_pipeline_delay = False
        self.enable_robot_body_subtraction = False
        
        # self.robot_description_path = None
        # self.camera_names = None
        # self.camera_tfs = None
        # self.agent_names = None
        # self.agent_tfs = None

        self.downsample = 5
        self.outlier_neighbours = 0

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

    def load_robot_urdf(self, robot_urdf_filepath:str):
        self.robot_description_path = robot_urdf_filepath
        self.robot_kinematics_chain = cph.kinematics.KinematicChain(self.robot_description_path)

    def setup(self):
        # set scene geometry properties
        min_bound = np.array(self.scene_bounds['min'])
        max_bound = np.array(self.scene_bounds['max'])
        self.voxel_min_bound = tuple(min_bound)
        self.voxel_max_bound = tuple(max_bound)
        
        if not np.all(np.abs(max_bound - min_bound) - (max_bound[0] - min_bound[0]) < 1e-6):
            raise ValueError("The min_bound and max_bound do not define a cube. Make it a cube!")

        self.scene_bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # set data buffers
        self.primitives_pos_gpu = None

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

    def _perform_robot_body_subtraction(self, pointclouds:dict, agent_tfs:dict, joint_states:dict):

        def get_mesh_from_joint_state(agent_tf, joint_state:dict):
            joint_state = joint_state['position']
            
            mesh = self.robot_kinematics_chain.forward_kinematics(joint_state)
            mesh = mesh.transform(agent_tf)
            

            return mesh

        # header:
        # seq: 58448
        # stamp:
        #     secs: 1740563297
        #     nsecs: 741921186
        # frame_id: ''
        # name:
        # - panda_joint1
        # - panda_joint2
        # - panda_joint3
        # - panda_joint4
        # - panda_joint5
        # - panda_joint6
        # - panda_joint7
        # - panda_finger_joint1
        # - panda_finger_joint2
        # position: [-0.01639333324892479, -0.2962552043524374, -0.021682949399173913, -1.9661479419484074, -0.051427240368392836, 1.6395320340216735, 0.6855043043063745, 0.03948273882269859, 0.03948273882269859]
        # velocity: [-0.00012541587041589654, -0.0005273575465614755, 0.0001572497684467771, 0.0004776579476729115, 0.0004162800392557667, 0.00040168241189413083, -0.0006871699583625779, 0.0, 0.0]
        # effort: [-0.06105424463748932, -20.465009689331055, -0.20073528587818146, 22.3988094329834, 0.7813851237297058, 2.3834891319274902, 0.11294838786125183, 0.0, 0.0]
        pass

    def _remove_outliers(self, pcd:cph.geometry.PointCloud):
        outlier_neighbours = self.outlier_neighbours
        outlier_std_ratio = 2.0
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=outlier_neighbours, std_ratio=outlier_std_ratio)
        return pcd

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

        return voxel_grid

    
    def _convert_voxels_to_primitives(self, voxel_grid:cph.geometry.VoxelGrid):
        start = time.time()
        voxels = voxel_grid.voxels.cpu()
        primitives_pos = np.array(list(voxels.keys()))

        if primitives_pos.size == 0:  # Handle empty voxel grid
            self.logger.warn("No voxels found in voxel grid")
            return None
        
        # Transfer data to GPU
        primitives_pos_gpu = cp.asarray(primitives_pos)
        offset = cp.asarray(voxel_grid.get_min_bound())
        voxel_size = cp.asarray(self.voxel_size)
        
        # Compute minimums for each column on GPU
        mins = cp.min(primitives_pos_gpu, axis=0)
        
        # Perform operations on GPU
        primitives_pos_gpu = primitives_pos_gpu - mins[None, :]
        primitives_pos_gpu = primitives_pos_gpu * voxel_size
        primitives_pos_gpu = primitives_pos_gpu + (offset + voxel_size/2)

        # save copy of primitives_pos_gpu
        self.primitives_pos_gpu = cuda.as_cuda_array(primitives_pos_gpu)
        
        # Transfer result back to CPU
        primitives_pos = cp.asnumpy(primitives_pos_gpu)

        if self.show_pipeline_delays:
            self.logger.info(f"Voxel2Primitives (CPU+GPU) [sec]: {time.time()-start}")
        return primitives_pos

    def run_pipeline(self, pointclouds:dict, camera_tfs:dict, agent_tfs:dict, joint_states:dict):
        # streamer/realsense gives pointclouds and tfs
        outlier_neighbours = self.outlier_neighbours
        enable_robot_body_subtraction = self.enable_robot_body_subtraction
        show_pipeline_delays = self.show_pipeline_delays
        show_total_pipeline_delay = self.show_total_pipeline_delay

        start = time.time()

        try:
            pointclouds = self._parse_pointclouds(pointclouds, camera_tfs)
            pointclouds = self._merge_pointclouds(pointclouds)
        except Exception as e:
            self.logger.error(troubleshoot.get_error_text(e))
            return None, None
        
        if enable_robot_body_subtraction:
            pointclouds = self._perform_robot_body_subtraction(pointclouds, agent_tfs, joint_states)

        if outlier_neighbours > 0:
            pointclouds = self._remove_outliers(pointclouds)

        if pointclouds is not None:
            voxel_grid = self._perform_voxelization(pointclouds)
            primitives_pos = self._convert_voxels_to_primitives(voxel_grid)
        else:
            primitives_pos = None

        if show_total_pipeline_delay:
            self.logger.info(f"Perception Pipeline (CPU+GPU) [sec]: {time.time()-start}")
        return primitives_pos
