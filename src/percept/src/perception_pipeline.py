import cupoch as cph
import numpy as np
from copy import deepcopy

import rospy
from sensor_msgs.msg import PointCloud2

import utils.troubleshoot as troubleshoot

class PerceptionPipeline():
    def __init__(self):
        # self.check_cuda() # add CUDA check
        pass

    def setup(self):
        min_bound, max_bound = np.array(self.scene_bounds['min']), np.array(self.scene_bounds['max'])
        self.bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    def read_and_process_obs(self, obs:dict):
        # loop to read and process pcds (to be parallelized)
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
                tf_matrix = obs[camera_name]['tf']
                pcd = pcd.transform(tf_matrix)

                # crop pcds according to defined scene bounds
                pcd = pcd.crop(self.bbox)

            except Exception as e:
                rospy.logerr(troubleshoot.get_error_text(e))

            obs[camera_name]['pcd'] = deepcopy(pcd)
        return obs

    def run(self, obs:dict):
        obs = self.read_and_process_obs(obs)
