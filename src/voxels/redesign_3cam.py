import os
import time 
import pickle

import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_white"



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
CAMERAS = ['cam1', 'cam2', 'cam3']

# Scene
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS    = (-1.5, -1.5, -1.5, 1.50, 1.5, 1.5)

# Agents
AGENTS = ['Franka0', 'Franka1']


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dataset Loading
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

# load data
DATASET_PATH = '../outputs/dualarms_3cam/dualarms_3cam_with_fk.pkl'
with open(DATASET_PATH, 'rb') as f:
    _obs = pickle.load(f)

def constrain_scene_bounds(data):
    global SCENE_BOUNDS

    nX, nY, nZ, X, Y, Z = SCENE_BOUNDS

    X_mask = (data[:,0] > nX) & (data[:,0]<X)
    Y_mask = (data[:,1] > nY) & (data[:,1]<Y)
    Z_mask = (data[:,2] > nZ) & (data[:,2]<Z)
    mask = (X_mask)&(Y_mask)&(Z_mask) 

    return data[mask]

def get_pcd_and_rgb(camera_obs:dict):
    pcd = camera_obs['pointcloud'].reshape(-2,3)
    rgb = camera_obs['rgb'].reshape(-2,3)
    return pcd, rgb

def get_pcd(camera_obs:dict):
    pcd = camera_obs['pointcloud'].reshape(-2,3)
    return pcd


obs = _obs[-1]
camera_obs = obs[CAMERAS[0]]
pcd, rgb = get_pcd_and_rgb(camera_obs)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

from scipy.spatial.transform import Rotation as R
import cupoch as cph

ROBOT_URDF_PATH = 'outputs/testdata/franka_panda_cupoch_3/panda.urdf'
URDF_JOINT_PREFIX = 'panda_joint'

path = os.path.join(ROOT, ROBOT_URDF_PATH)
kin = cph.kinematics.KinematicChain(path)


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

def get_mesh_using_forward_kinematics(
    joints_positions:np.array,
    global_position:np.array,
    global_rotation:np.array
):
    global URDF_JOINT_PREFIX

    # create joint_map, used for fk on urdf model
    joint_map = {'%s%d' % (URDF_JOINT_PREFIX, i+1): val for i, val \
                in enumerate(joints_positions)} 

    # manual offset, to be removed 
    # offset = np.array([0.0, 0.0, -0.06999997])
    offset = np.array([0.0, 0.0, 0.0])


    # create transformation matrix
    tf_matrix = get_tf_matrix(global_position+offset, global_rotation)
    
    # do forward kinematics
    poses = kin.forward_kinematics(joint_map, tf_matrix)
    
    # store mesh geometries
    meshes = kin.get_transformed_visual_geometry_map(poses)
    
    return list(meshes.values())



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def get_pcds_for_registration(obs):
    global CAMERAS

    pcds = list()
    for CAMERA in CAMERAS:
        camera_obs = obs[CAMERA]
        pcd, rgb = get_pcd_and_rgb(camera_obs)
        pcd = constrain_scene_bounds(pcd)
        pcds.append(pcd)

    return pcds

pcds = get_pcds_for_registration(obs)



def do_point_cloud_registration(pcds):
    
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


    elapsed_time = time.time() - start
    print("ICP (GPU) [sec]:", elapsed_time) # adding outlier removal adds ~25ms

    # cph.visualization.draw_geometries([source_gpu+target_gpu])

    return source_gpu+target_gpu


pcd = do_point_cloud_registration(pcds)



def do_point_cloud_registration(pcds):
    assert len(pcds) > 1
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
    source_gpu = register(source_gpu, target_gpu)

    # merge all other pcds with new source
    for pcd in pcds[2:]:
        target_gpu = cph.geometry.PointCloud(
            cph.utility.HostVector3fVector(pcd)
        )
        source_gpu = register(source_gpu, target_gpu)


    return source_gpu


pcd = do_point_cloud_registration(pcds)


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def get_bb_from_mesh(
    mesh:cph.geometry.TriangleMesh
) -> cph.geometry.AxisAlignedBoundingBox:

    AABB = mesh.get_axis_aligned_bounding_box()
    if AABB.is_empty():
        print('empty mesh detected')
        return None       
    else:
        return AABB
    

def perform_rbs_on_pointcloud_using_bb(obs, pcd):

    n = len(pcd.points)
    masks = list()
    meshes_list = list()
    for agent in AGENTS:
        print('\nAgent: %s' % agent)
        agent_obs = obs[agent]
        
        # get all meshes of the agent using fk
        meshes = get_mesh_using_forward_kinematics(
            agent_obs['joint_pos'],
            agent_obs['global_pos'],
            agent_obs['global_ang'],
        )
        meshes_list += meshes

        # find points within BBs and create masks
        start = time.time()
        for mesh in meshes:
            bb = get_bb_from_mesh(mesh)

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
    ), meshes_list

rbs_pcd, meshes = perform_rbs_on_pointcloud_using_bb(obs, pcd)


cph.visualization.draw_geometries([rbs_pcd]+meshes)