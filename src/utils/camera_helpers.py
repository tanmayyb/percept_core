import numpy as np
from transforms3d.euler import euler2quat, quat2mat
from geometry_msgs.msg import TransformStamped

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



# import cupoch as cph
# def create_observation(msg, static_configs:dict, log_and_kill:bool=False) -> dict:
#     # https://ksimek.github.io/2012/08/22/extrinsic/
#     # read message and create oveservation

#     width = msg.width
#     height = msg.height

#     # read depth image
#     def get_numpy_dtype(encoding):
#         """Map ROS encoding to numpy dtype."""
#         if encoding in ['mono8', '8UC1']:
#             return np.uint8
#         elif encoding in ['mono16', '16UC1']:
#             return np.uint16
#         elif encoding in ['rgb8', 'bgr8', '8UC3']:
#             return np.uint8
#         elif encoding in ['16UC3']:
#             return np.uint16
#         else:
#             raise ValueError(f"Unsupported encoding: {encoding}")
#     np_img = np.frombuffer(msg.data, get_numpy_dtype(msg.encoding)).reshape(height, width)
#     depth_image = cph.geometry.Image(np_img.astype(np.float32))


#     intrinsics = cph.camera.PinholeCameraIntrinsic()
#     intrinsics.set_intrinsics(
#         msg.width,
#         msg.height,
#         static_configs['intrinsics']['fx'],
#         static_configs['intrinsics']['fy'],
#         static_configs['intrinsics']['cx'],
#         static_configs['intrinsics']['cy']
#     )

#     # create extrinsics_matrix
#     extrinsics = create_extrinsic_matrix(static_configs['pose'])

#     obs = dict(
#         depth_image = depth_image,
#         intrinsics = intrinsics,
#         extrinsics = extrinsics
#     )


#     if log_and_kill:
#         # Save observation dictionary to file
#         import pickle
#         import sys
#         import rclpy
        
#         rclpy.get_logger('camera_helpers').info('logging and killing...')
#         try:
#             # Save data to pickle file
#             data_to_save = {
#                 'depth_image': np_img,
#                 'width': msg.width,
#                 'height': msg.height,
#                 'static_configs': static_configs,
#                 'extrinsics': extrinsics
#             }
            
#             with open('/tmp/camera_data.pickle', 'wb') as f:
#                 pickle.dump(data_to_save, f)
#         except Exception as e:
#             rclpy.get_logger('camera_helpers').error(f'{e}')

#         rclpy.shutdown()
#         sys.exit(0)

#     return obs