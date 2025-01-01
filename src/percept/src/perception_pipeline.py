import cupoch as cph
import numpy as np
from copy import deepcopy

import rospy
from sensor_msgs.msg import PointCloud2

import utils.troubleshoot as troubleshoot

class PerceptionPipeline():
    def __init__(self):
        # self.check_cuda() # add CUDA check
        self.obs = None

    def setup(self):
        min_bound, max_bound = np.array(self.scene_bounds['min']), np.array(self.scene_bounds['max'])
        self.bbox = cph.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    def read_and_process_pcds(self, msg: PointCloud2):
        # loop to read and process pcds (to be parallelized)
        for camera_name in self.camera_names:
            try:
                # read pcds
                pcd = cph.geometry.PointCloud()
                temp = cph.io.create_from_pointcloud2_msg(
                    msg.data, cph.io.PointCloud2MsgInfo.default(msg.width, msg.point_step)
                )
                pcd.points = temp.points

                # transform pcds to world frame           
                tf_matrix = self.cameras[camera_name]['tf']
                pcd = pcd.transform(tf_matrix)

                # crop pcds according to defined scene bounds
                pcd = pcd.crop(self.bbox)

            except Exception as e:
                rospy.logerror(troubleshoot.get_error_text(e))

            self.obs[camera_name]['pcl'] = deepcopy(pcd)

            rospy.loginfo('completed!')
    
    def register_pcds(self):
        pass
    def do_rbs(self):
        pass
    def voxelize_pcds(self):
        pass

    def run(self, obs:dict):
        self.read_and_process_pcds(obs)
        # self.register_pcds()
        # self.do_rbs()
        # self.voxelize_pcds()

