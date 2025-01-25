import cupoch as cph
import numpy as np
import cupy as cp

import rospy
import time
import utils.troubleshoot as troubleshoot

import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from kernels.radial_distance_compute import compute_radial_distances, compute_radial_distance_vectors

class PerceptionPipeline():
    def __init__(self):
        self.check_cuda()
        # Create process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        # Create thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    def check_cuda(self):
        """Check if CUDA is available using nvidia-smi"""
        try:
            output = subprocess.check_output(["nvidia-smi"])
            rospy.loginfo("CUDA is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            rospy.logerr("CUDA is not available - nvidia-smi command failed")
            raise RuntimeError("CUDA is required for this pipeline")

    def setup(self):
        # set scene props
        min_bound, max_bound = np.array(self.scene_bounds['min']), np.array(self.scene_bounds['max'])
        self.scene_bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # set voxel props
        cubic_size = self.cubic_size
        voxel_resolution = self.voxel_resolution
        self.voxel_size = cubic_size/voxel_resolution
        self.voxel_min_bound = (-cubic_size/2.0, -cubic_size/2.0, -cubic_size/2.0)
        self.voxel_max_bound = (cubic_size/2.0, cubic_size/2.0, cubic_size/2.0)

    def _process_single_pointcloud(self, camera_name:str, msg, tf_matrix=None, downsample=False):
        try:
            # load pointcloud from ROS msg
            pcd = cph.geometry.PointCloud()
            temp = cph.io.create_from_pointcloud2_msg(
                msg.data, cph.io.PointCloud2MsgInfo.default_dense(
                    msg.width, msg.height, msg.point_step)
            )
            pcd.points = temp.points

            # if tf_matrix available, transform to world-frame
            if tf_matrix is not None:
                pcd = pcd.transform(tf_matrix)

            # crop pointcloud according to scene bounds
            pcd = pcd.crop(self.scene_bbox)
            
            if downsample:
                every_n_points = 3
                pcd = pcd.uniform_down_sample(every_n_points)

            return pcd
        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))


    def parse_pointclouds(self, pointclouds:dict, tfs:dict, use_sim=False, downsample:bool=False, log_performance:bool=False):
        # loop to read and process pcds (to be parallelized)
        start = time.time()

        # pointcloud assertion check
        try:
            assert pointclouds is not None
        except Exception as e:
            rospy.logerr(troubleshoot.get_error_text(e))

        # Process point clouds in parallel using threads
        futures = list()
        for camera_name in self.camera_names:
            futures.append(
                self.thread_pool.submit(
                    self._process_single_pointcloud,
                    camera_name,
                    pointclouds[camera_name],
                    None if use_sim else tfs.get(camera_name),
                    downsample
                )
            )
        
        # Collect results
        processed_pointclouds = dict()
        for camera_name, future in zip(self.camera_names, futures):
            try:
                processed_pointclouds[camera_name] = future.result()
            except Exception as e:
                rospy.logerr(f"Error processing {camera_name}: {troubleshoot.get_error_text(e)}")
        
        if log_performance:
            rospy.loginfo(f"PCD Processing (CPU+GPU) [sec]: {time.time()-start}")

        return processed_pointclouds
    
    def merge_pointclouds(self, pointclouds:dict, log_performance:bool=False):
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

        if log_performance:
            rospy.loginfo(f"Registration (GPU) [sec]: {time.time()-start}")
        return merged_gpu

    def perform_robot_body_subtraction(self):
        pass


    def perform_voxelization(self, pcd:cph.geometry.PointCloud, log_performance:bool=False):
        start = time.time()
        voxel_grid = cph.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=self.voxel_size,
            min_bound=self.voxel_min_bound,
            max_bound=self.voxel_max_bound,
        )
        if log_performance:
            rospy.loginfo(f"Voxelization (GPU) [sec]: {time.time()-start}")

        return voxel_grid

    
    def convert_voxels_to_primitives(self, voxel_grid:cph.geometry.VoxelGrid, log_performance:bool=False):
        start = time.time()
        voxels = voxel_grid.voxels.cpu()
        primitives_pos = np.array(list(voxels.keys()))

        if primitives_pos.size == 0:  # Handle empty voxel grid
            rospy.logwarn("No voxels found in voxel grid")
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
        
        # Transfer result back to CPU
        primitives_pos = cp.asnumpy(primitives_pos_gpu)

        if log_performance:
            rospy.loginfo(f"Voxel2Primitives (CPU+GPU) [sec]: {time.time()-start}")
        return primitives_pos

    def compute_radial_distance_vectors(self, primitives_pos:np.ndarray, log_performance:bool=False):
        start = time.time()
        # NOTE: temp fix NOTE: IMPORTANT: add agent location in future!
        agent_location = np.array([0.0, 0.0, 0.0]) # anchor point
        search_radius = 0.5
        # TODO: add sphere radius to include only primitives which touch the search shell
        # voxel_size = self.voxel_size

        distance_vectors = compute_radial_distance_vectors(agent_location, primitives_pos, search_radius)
        if log_performance:
            rospy.loginfo(f"Compute Distance Vectors (CPU+GPU) [sec]: {time.time()-start}")
        return distance_vectors

    def run_pipeline(self, pointclouds:dict, tfs:dict, use_sim=False, log_performance:bool=False):
        # streamer/realsense gives pointclouds and tfs
        log_performance = False
        start = time.time()
        pointclouds = self.parse_pointclouds(pointclouds, tfs, use_sim=use_sim, downsample=True, log_performance=log_performance) # downsample increases performance
        merged_pointclouds = self.merge_pointclouds(pointclouds, log_performance=log_performance)
        # pointclouds = self.perform_robot_body_subtraction()
        voxel_grid = self.perform_voxelization(merged_pointclouds, log_performance=log_performance)
        primitives_pos = self.convert_voxels_to_primitives(voxel_grid, log_performance=log_performance)
        primitives_distance_vectors = self.compute_radial_distance_vectors(primitives_pos, log_performance=log_performance)
        
        log_performance = True
        if log_performance:
            rospy.loginfo(f"Perception Pipeline (CPU+GPU) [sec]: {time.time()-start}")
        # return primitives_pos

        # NOTE: temp fix
        return primitives_pos, primitives_distance_vectors


