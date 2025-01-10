import cupoch as cph
import numpy as np
from copy import deepcopy

import rospy
from sensor_msgs.msg import PointCloud2

import time
import utils.troubleshoot as troubleshoot

import subprocess

class PerceptionPipeline():
    def __init__(self):
        self.check_cuda()

    def setup(self):

        # set scene props
        min_bound, max_bound = np.array(self.scene_bounds['min']), np.array(self.scene_bounds['max'])
        self.bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # set voxel props
        cubic_size = self.cubic_size
        voxel_resolution = self.voxel_resolution
        self.voxel_size = cubic_size/voxel_resolution
        self.voxel_min_bound = (-cubic_size/2.0, -cubic_size/2.0, -cubic_size/2.0)
        self.voxel_max_bound = (cubic_size/2.0, cubic_size/2.0, cubic_size/2.0)

    def check_cuda(self):
        """Check if CUDA is available using nvidia-smi"""
        try:
            output = subprocess.check_output(["nvidia-smi"])
            rospy.loginfo("CUDA is available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            rospy.logerr("CUDA is not available - nvidia-smi command failed")
            raise RuntimeError("CUDA is required for this pipeline")

    def read_and_process_obs(self, obs:dict, log_performance:bool=False, sim=False, downsample=False):
        # loop to read and process pcds (to be parallelized)
        start = time.time()
        for camera_name in self.camera_names:
            try:
                msg = obs[camera_name]['pcd']

                # read pcds
                pcd = cph.geometry.PointCloud()
                temp = cph.io.create_from_pointcloud2_msg(
                    msg.data, cph.io.PointCloud2MsgInfo.default_dense(
                        msg.width, msg.height, msg.point_step)
                )
                pcd.points = temp.points

                # transform pcds to world frame   
                if not(sim):        
                    tf_matrix = obs[camera_name]['tf']
                    pcd = pcd.transform(tf_matrix)

                # crop pcds according to defined scene bounds
                pcd = pcd.crop(self.bbox)

                # downsample pcds
                if downsample:
                    every_n_points = 3
                    pcd = pcd.uniform_down_sample(every_n_points)

                # replace pcd object
                obs[camera_name]['pcd'] = deepcopy(pcd)

            except Exception as e:
                rospy.logerr(troubleshoot.get_error_text(e))
        
        if log_performance:
            rospy.loginfo(f"PCD Processing (CPU+GPU) [sec]: {time.time()-start}")

        return obs
    
    def perform_pcd_registration(self, obs:dict, log_performance:bool=False):
        # supposed to perform registration

        start = time.time()


        def register(
            source_gpu:cph.geometry.PointCloud, 
            target_gpu:cph.geometry.PointCloud,
            remove_outliers=False
        ) -> cph.geometry.PointCloud:
            target_gpu.estimate_normals()
            threshold = 0.02 # what does this do?
            trans_init = np.asarray(
                [[1.0, 0.0, 0.0, 0.0], 
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0], 
                 [0.0, 0.0, 0.0, 1.0]])

            # register pointclouds
            register_icp = cph.registration.registration_icp(
                source_gpu,
                target_gpu,
                threshold,
                trans_init.astype(np.float32),
                cph.registration.TransformationEstimationPointToPlane(),
            )

            source_gpu.transform(register_icp.transformation)

            if remove_outliers:
                # remove outliers
                NEIGHBOURS = 2
                source_gpu, ind = source_gpu.remove_statistical_outlier(nb_neighbors=NEIGHBOURS, std_ratio=2.0)
                target_gpu, ind = target_gpu.remove_statistical_outlier(nb_neighbors=NEIGHBOURS, std_ratio=2.0)
            return source_gpu+target_gpu

        def dirty_merge(
            source_gpu:cph.geometry.PointCloud,
            target_gpu:cph.geometry.PointCloud
        ):
            return source_gpu+target_gpu



        source_gpu = obs[self.camera_names[0]]['pcd']
        # source_gpu.estimate_normals()
        merged_gpu = source_gpu
        if len(self.camera_names)>1:
            target_gpu = obs[self.camera_names[1]]['pcd']
            # merged_gpu = register(merged_gpu, 
            # target_gpu)    
            merged_gpu = dirty_merge(merged_gpu, target_gpu)
        if len(self.camera_names)>2:
            for camera_name in self.camera_names[2:]:
                target_gpu = obs[camera_name]['pcd']
                # merged_gpu = register(merged_gpu, 
                # target_gpu)
                merged_gpu = dirty_merge(merged_gpu, target_gpu)

        if log_performance:
            rospy.loginfo(f"Registration (GPU) [sec]: {time.time()-start}")
        return merged_gpu


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
        offset = voxel_grid.get_min_bound()
        voxel_size = self.voxel_size
        primitives_pos = np.array(list(voxels.keys()))
        primitives_pos = (primitives_pos - np.array([
                        min(primitives_pos[:,0]), 
                        min(primitives_pos[:,1]), 
                        min(primitives_pos[:,2])]))
        primitives_pos = primitives_pos*voxel_size
        primitives_pos += (offset + voxel_size/2)

        if log_performance:
            rospy.loginfo(f"Voxel2Primitives (CPU) [sec]: {time.time()-start}")
        return primitives_pos


    def run(self, obs:dict, log_performance:bool=False, sim=False):
        
        log_performance = True

        start = time.time()
        obs = self.read_and_process_obs(obs, sim=sim, downsample=False, log_performance=True)
        joint_pcd = self.perform_pcd_registration(obs, log_performance=True)
        # do rbs
        voxel_grid = self.perform_voxelization(joint_pcd, log_performance=True)
        primitive_pos = self.convert_voxels_to_primitives(voxel_grid, log_performance=True)

        if log_performance:
            rospy.loginfo(f"Perception Pipeline (CPU+GPU) [sec]: {time.time()-start}")

        return primitive_pos