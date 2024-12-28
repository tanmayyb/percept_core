import cupoch as cph
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import os


# Root of Project
ROOT = '/home/dev/ws_percept/src'

# # Cameras
# IMAGE_SIZE  = 240
# CAMERAS = ['cam1', 'cam2', 'cam3']

# Scene
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS    = (-1.5, -1.5, -1.5, 1.50, 1.5, 1.5)

ARM_NAME = 'panda'

# Agents
AGENTS = [f'{ARM_NAME}0', f'{ARM_NAME}1']

SHOW_CPH_VIZ = False

ROBOT_URDF_PATH = '../assets/franka_panda_cupoch/panda.urdf'
URDF_JOINT_PREFIX = 'panda_joint'

CUBIC_SIZE = 2.0
VOXEL_RESOLUTION = 30.0 # = 100.0
SPHERICAL_RADIUS = 0.03



class PerceptionPipeline():
    def __init__(self):
        # self.observations = None
        # self.obs = None
        # self.pcds = None
        # self.pcd = None
        # self.kin = None
        # self.meshes_list = None
        # self.voxels = None
        # self.voxel_size = None
        # self.v2s_positions = None
        # self.spheres = None
        pass


    # def load_dataset(self, DATASET_PATH) -> None:
    #     with open(DATASET_PATH, 'rb') as f:
    #         self.observations = pickle.load(f)

    # def select_obs(self, n:int=1) -> None:
    #     self.obs = self.observations[n]
    #     self.get_pcds_for_registration()

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Get Pointclouds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def constrain_scene_bounds(self, data) -> np.array:
        global SCENE_BOUNDS
        nX, nY, nZ, X, Y, Z = SCENE_BOUNDS
        X_mask = (data[:,0] > nX) & (data[:,0]<X)
        Y_mask = (data[:,1] > nY) & (data[:,1]<Y)
        Z_mask = (data[:,2] > nZ) & (data[:,2]<Z)
        mask = (X_mask)&(Y_mask)&(Z_mask) 
        return data[mask]

    def get_pcd_and_rgb(self, camera_obs:dict) -> tuple:
        pcd = camera_obs['pointcloud'].reshape(-2,3)
        rgb = camera_obs['rgb'].reshape(-2,3)
        return pcd, rgb
    
    def get_pcds_for_registration(self) -> None:
        global CAMERAS
        obs = self.obs
        assert obs is not None, "Did you select the obs?"

        pcds = list()
        for CAMERA in CAMERAS:
            camera_obs = obs[CAMERA]
            pcd, rgb = self.get_pcd_and_rgb(camera_obs)
            pcd = self.constrain_scene_bounds(pcd)
            pcds.append(pcd)

        # return pcds
        self.pcds = pcds

    def do_point_cloud_registration(
        self
    ) -> None: #cph.geometry.PointCloud:

        pcds = self.pcds
        assert pcds is not None, "Did you get the pcds?"
        assert len(pcds) > 1, "Are you using only 1 camera?"

        def register(
            source_gpu:cph.geometry.PointCloud, 
            target_gpu:cph.geometry.PointCloud,
        ) -> cph.geometry.PointCloud:

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


            elapsed_time = time.time() - start
            print("ICP (GPU) [sec]:", elapsed_time) # adding outlier removal adds ~25ms
        
            return source_gpu+target_gpu


        # load first and second pcd
        source_gpu = cph.geometry.PointCloud(
            cph.utility.HostVector3fVector(pcds[0])
        )
        target_gpu = cph.geometry.PointCloud(
            cph.utility.HostVector3fVector(pcds[1])
        )
        # register pcds
        source_gpu = register(source_gpu, target_gpu)

        # merge all other pcds with new source
        for pcd in pcds[2:]:
            target_gpu = cph.geometry.PointCloud(
                cph.utility.HostVector3fVector(pcd)
            )
            source_gpu = register(source_gpu, target_gpu)
        # return source_gpu
        self.pcd = source_gpu

    # # def run(self):
    # #     global DATASET_SAVE_FILEPATH
    # #     self.load_dataset(DATASET_SAVE_FILEPATH)
    # #     self.select_obs(1)
    # #     self.do_point_cloud_registration()
    # #     self.perform_rbs_on_pointcloud_using_bb()
    # #     self.convert_pointcloud_to_voxels()
    # #     self.voxel2sphere()

