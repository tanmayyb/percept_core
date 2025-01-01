import numpy as np
from tf.transformations import quaternion_matrix

def create_tf_matrix_from_msg(transform_msg):
    """
    Create a 4x4 transformation matrix using tf.transformations.
    
    :param transform_msg: TransformStamped message containing translation and rotation.
    :return: 4x4 numpy array representing the transformation matrix.
    """
    # Extract translation
    translation = np.array([
        transform_msg.transform.translation.x,
        transform_msg.transform.translation.y,
        transform_msg.transform.translation.z
    ])
    
    # Extract quaternion
    quaternion = [
        transform_msg.transform.rotation.x,
        transform_msg.transform.rotation.y,
        transform_msg.transform.rotation.z,
        transform_msg.transform.rotation.w
    ]
    
    # Generate the rotation matrix from quaternion
    rotation_matrix = quaternion_matrix(quaternion)  # 4x4 matrix
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


def create_intrinsic_matrix( intrinsics_config):
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
#         import rospy
        
#         rospy.loginfo(f'logging and killing...')
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
#             rospy.logerr(f'{e}')

#         rospy.signal_shutdown("Shutting down node")
#         sys.exit(0)

#     return obs