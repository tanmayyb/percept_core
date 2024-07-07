import os
import time 
import pickle

from copy import deepcopy



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Project Description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# Root of Project
ROOT = '/home/dev/ws_percept/src'

# Cameras
IMAGE_SIZE  = 240
CAMERAS = ['front', 'back']

# Scene
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS    = (-1.5, -1.5, -1.5, 1.50, 1.5, 1.5)

# Agents
AGENTS = ['Franka0', 'Franka1']

# URDF model
ROBOT_URDF_PATH = 'outputs/testdata/franka_panda_cupoch_2/panda.urdf'
URDF_JOINT_PREFIX = 'panda_joint'

# Pipeline Visualizations
SHOW_VISUALIZATION = True


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dataset Loading
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# load data
DATASET_PATH = 'outputs/dualarms_2cam/dualarms_2cam_with_fk.pkl'
with open(DATASET_PATH, 'rb') as f:
    _obs = pickle.load(f)


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pipeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import cupoch as cph
import time

class Pipeline():
    def __init__(
        self,
    ):
        pass

    """PCD, Mesh Methos"""
    @staticmethod
    def constrain_scene_bounds(
        data:np.array
    ) -> np.array:
        global SCENE_BOUNDS

        nX, nY, nZ, X, Y, Z = SCENE_BOUNDS
        X_mask = (data[:,0] > nX) & (data[:,0]<X)
        Y_mask = (data[:,1] > nY) & (data[:,1]<Y)
        Z_mask = (data[:,2] > nZ) & (data[:,2]<Z)
        mask = (X_mask)&(Y_mask)&(Z_mask) 

        return data[mask]

    @staticmethod
    def get_pcd_and_rgb(camera_obs:dict) -> tuple:
        pcd = camera_obs['pointcloud'].reshape(-2,3)
        rgb = camera_obs['rgb'].reshape(-2,3)
        return pcd, rgb

    @staticmethod
    def get_pcd(camera_obs:dict) -> np.array:
        pcd = camera_obs['pointcloud'].reshape(-2,3)
        return pcd

    @staticmethod
    def get_tf_matrix(
        translation:np.array,
        rotation:np.array, 
    ) -> np.array:
        r = R.from_rotvec(rotation)
        r = r.as_matrix()

        tf = np.eye(4)
        tf[:3,:3] = r
        tf[:3,3] = translation

        return tf

    @staticmethod
    def load_urdf_kinematics_chain() -> cph.kinematics.KinematicChain:
        global ROOT, ROBOT_URDF_PATH
        path = os.path.join(ROOT, ROBOT_URDF_PATH)
        kin = cph.kinematics.KinematicChain(path)
        return kin
    
    def get_mesh_using_forward_kinematics(
        self,
        kin:cph.kinematics.KinematicChain,
        joints_positions:np.array,
        global_position:np.array,
        global_rotation:np.array
    ) -> list:
        global URDF_JOINT_PREFIX

        # create joint_map, used for fk on urdf model
        joint_map = {'%s%d' % (URDF_JOINT_PREFIX, i+1): val for i, val \
                    in enumerate(joints_positions)} 

        # manual offset, to be removed 
        offset = np.array([0.0, 0.0, -0.06999997])

        # create transformation matrix
        tf_matrix = self.get_tf_matrix(global_position+offset, global_rotation)
        
        # do forward kinematics
        poses = kin.forward_kinematics(joint_map, tf_matrix)
        
        # store mesh geometries
        meshes = kin.get_transformed_visual_geometry_map(poses)
        
        return list(meshes.values())
    

    """PCR"""
    def get_pcds_for_registration(self, obs:dict) -> list:
        global CAMERAS
        pcds = list()
        camera_obs = obs[CAMERAS[0]]
        pcd, rgb = self.get_pcd_and_rgb(camera_obs)
        pcd = self.constrain_scene_bounds(pcd)
        pcds.append(pcd)

        camera_obs = obs[CAMERAS[1]]
        pcd, rgb = self.get_pcd_and_rgb(camera_obs)
        pcd = self.constrain_scene_bounds(pcd)
        pcds.append(pcd)
        
        return pcds
    
    @staticmethod
    def do_point_cloud_registration(
        pcds:list
    )->cph.geometry.PointCloud:
        global SHOW_VISUALIZATION

        # load source and target pointcloud
        source_gpu = cph.geometry.PointCloud(
            cph.utility.HostVector3fVector(pcds[0])
        )
        target_gpu = cph.geometry.PointCloud(
            cph.utility.HostVector3fVector(pcds[1])
        )

        threshold = 0.02 # what does this do?
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        # register pointclouds
        start = time.time()
        reg_p2p = cph.registration.registration_icp(
            source_gpu,
            target_gpu,
            threshold,
            trans_init.astype(np.float32),
            cph.registration.TransformationEstimationPointToPlane(),
        )

        source_gpu.transform(reg_p2p.transformation)


        # remove outliers
        NEIGHBOURS = 2
        source_gpu, ind = source_gpu.remove_statistical_outlier(nb_neighbors=NEIGHBOURS, std_ratio=2.0)
        target_gpu, ind = target_gpu.remove_statistical_outlier(nb_neighbors=NEIGHBOURS, std_ratio=2.0)

        pcd = source_gpu+target_gpu
        elapsed_time = time.time() - start
        print("ICP (GPU) [sec]:", elapsed_time) # adding outlier removal adds ~25ms

        if SHOW_VISUALIZATION:
            cph.visualization.draw_geometries([pcd])

        return pcd 
    
    """RBS"""
    @staticmethod
    def get_bb_from_mesh(
        mesh:cph.geometry.TriangleMesh
    ) -> cph.geometry.AxisAlignedBoundingBox:

        AABB = mesh.get_axis_aligned_bounding_box()
        if AABB.is_empty():
            print('empty mesh detected')
            return None       
        else:
            return AABB

    def perform_rbs_on_pointcloud_using_bb(
        self, 
        obs:dict, 
        pcd:cph.geometry.PointCloud,
    ) -> cph.geometry.PointCloud:
        global AGENTS

        n = len(pcd.points)
        masks = list()
        kin = self.load_urdf_kinematics_chain()
        for agent in AGENTS:
            print('\nAgent: %s' % agent)
            agent_obs = obs[agent]
            
            # get all meshes of the agent using fk
            meshes = self.get_mesh_using_forward_kinematics(
                kin,
                agent_obs['joint_pos'],
                agent_obs['global_pos'],
                agent_obs['global_ang'],
            )

            # find points within BBs and create masks
            start = time.time()
            for mesh in meshes:
                bb = self.get_bb_from_mesh(mesh)

                if bb is not None:
                    mask = np.zeros(n, dtype=bool)

                    indices_points_within_bb = np.asarray(
                        bb.get_point_indices_within_bounding_box(
                            pcd.points
                        ).cpu()
                    )
                    if len(indices_points_within_bb)>0:
                        mask[indices_points_within_bb] = True
                        masks.append(mask)

            elapsed_time = time.time() - start
            print("RBS (CPU+GPU) [sec]:", elapsed_time)

        mask = np.column_stack(tuple(masks)).any(axis=1)
        mask = ~mask # inverting mask to get pcd which were not within any BBs
        
        pcd_arr = np.asarray(pcd.points.cpu())
        return  cph.geometry.PointCloud(
            cph.utility.HostVector3fVector(pcd_arr[mask])
        )
    
    
    """Voxels"""
    @staticmethod
    def convert_pointcloud_to_voxels(
        pcd:cph.geometry.PointCloud
    ) -> cph.geometry.VoxelGrid:
        global SHOW_VISUALIZATION
        cubic_size = 2.0
        voxel_resolution = 100.0

        # create voxel grid
        start = time.time()
        voxels = cph.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=cubic_size / voxel_resolution,
            min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
            max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2),
        )
        elapsed_time = time.time() - start
        print("Voxelization (GPU) [sec]:", elapsed_time) # adding outlier removal adds ~25ms

        if SHOW_VISUALIZATION:
            cph.visualization.draw_geometries([voxels])

        return voxels

    def forward(self):
        global _obs

        # get pcd
        obs = _obs[-1]
        pcds = self.get_pcds_for_registration(obs)

        # PCR
        pcd = self.do_point_cloud_registration(pcds)

        # RBS
        rbs_pcd = self.perform_rbs_on_pointcloud_using_bb(obs, pcd)

        # Voxels
        voxels = self.convert_pointcloud_to_voxels(rbs_pcd)

       
pipeline = Pipeline()
pipeline.forward()