import numpy as np
from transforms3d.euler import euler2quat, quat2mat
from geometry_msgs.msg import TransformStamped
import cupy as cp

def create_tf_msg_from_xyzrpy(
        child_frame:str, 
        x:float, y:float, z:float,
        a:float, b:float, g:float, 
        frame_id='world'):
    # https://robotics.stackexchange.com/questions/53148/quaternion-transformations-in-python
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = frame_id
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.translation.x = x
    tf_msg.transform.translation.y = y
    tf_msg.transform.translation.z = z
    
    quaternion = euler2quat(a, b, g, 'rxyz')
    # added 90deg to offset the camera NOA convention
    tf_msg.transform.rotation.x = quaternion[1]
    tf_msg.transform.rotation.y = quaternion[2]
    tf_msg.transform.rotation.z = quaternion[3]
    tf_msg.transform.rotation.w = quaternion[0]

    return tf_msg

def create_tf_matrix_from_msg(transform_msg):
    """
    Create a 4x4 transformation matrix using transforms3d.
    
    :param transform_msg: TransformStamped message containing translation and rotation.
    :return: 4x4 numpy array representing the transformation matrix.
    """
    # Extract translation
    translation = np.array([
        transform_msg.transform.translation.x,
        transform_msg.transform.translation.y,
        transform_msg.transform.translation.z
    ])
    
    # Extract quaternion (w,x,y,z)
    quaternion = [
        transform_msg.transform.rotation.w,
        transform_msg.transform.rotation.x,
        transform_msg.transform.rotation.y,
        transform_msg.transform.rotation.z
    ]
    
    # Generate the rotation matrix from quaternion
    rotation_matrix = np.eye(4)
    rotation_matrix[:3,:3] = quat2mat(quaternion)
    # Set the translation
    rotation_matrix[:3, 3] = translation

    return rotation_matrix


def create_tf_matrix_from_euler(pose_config):
    roll = pose_config['orientation']['roll']
    pitch = pose_config['orientation']['pitch']
    yaw = pose_config['orientation']['yaw']
    
    # Convert RPY to rotation matrix
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combine rotations
    R = Rz @ Ry @ Rx

    # get position vector
    position = np.array([
        pose_config['position']['x'],
        pose_config['position']['y'],            
        pose_config['position']['z'],
    ])

    # Create 4x4 extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3,:3] = R
    extrinsic_matrix[:3,3] = position
    return extrinsic_matrix

    # x, y, z = pose_config['position']['x'], \
    #     pose_config['position']['y'], \
    #     pose_config['position']['z']
    # roll, pitch, yaw = pose_config['orientation']['roll'], \
    #     pose_config['orientation']['pitch'], \
    #     pose_config['orientation']['yaw']
    # x, y, z = [cp.asarray(pose_config['position'][k]) for k in ['x', 'y', 'z']]
    # roll, pitch, yaw = [cp.asarray(pose_config['orientation'][k]) for k in ['roll', 'pitch', 'yaw']]

    # extrinsic_matrix = cp.eye(4)  # Create a 4x4 identity matrix
    
    # # Create rotation matrices
    # Rx = cp.array([[1, 0, 0],
    #                 [0, cp.cos(roll), -cp.sin(roll)],
    #                 [0, cp.sin(roll), cp.cos(roll)]])
    
    # Ry = cp.array([[cp.cos(pitch), 0, cp.sin(pitch)],
    #                 [0, 1, 0],
    #                 [-cp.sin(pitch), 0, cp.cos(pitch)]])
    
    # Rz = cp.array([[cp.cos(yaw), -cp.sin(yaw), 0],
    #                 [cp.sin(yaw), cp.cos(yaw), 0],
    #                 [0, 0, 1]])
    
    # # Combine rotations
    # R = Rz @ Ry @ Rx
    
    # extrinsic_matrix[:3, :3] = R
    # extrinsic_matrix[:3, 3] = cp.array([x, y, z])

    # return extrinsic_matrix.get()



def create_intrinsic_matrix(intrinsics_config):
    fx = intrinsics_config['fx']
    fy = intrinsics_config['fy'] 
    cx = intrinsics_config['cx']
    cy = intrinsics_config['cy']

    # Create 3x3 intrinsic matrix
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy], 
        [0, 0, 1]
    ])
    return intrinsic_matrix

