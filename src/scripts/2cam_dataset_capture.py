import os, sys, logging, time, pickle
import numpy as np
from pyrep import PyRep
from pyrep.objects import VisionSensor, Dummy


os.environ["DISPLAY"] = ":0"
os.environ["PYOPENGL_PLATFORM"] = "egl"

SIMULATION_TIME = 5.0
HEADLESS    = True
IMAGE_SIZE  = 128

# launch scene
pr = PyRep()
pr.launch(
    './scenes/test_scene_reinforcement_learning_env.ttt', 
    headless=HEADLESS,
) 
pr.start() 


# The name of the sensor in the vrep scene
cam = VisionSensor('front_rgbd')  
cam.set_resolution([IMAGE_SIZE,IMAGE_SIZE])




class Observer():
    def __init__(
        self, 
        cam:VisionSensor
    ):
        self.cam    = cam
        self.obs    = list()

    def add_new_observation(self) -> None:
        rgb = self.cam.capture_rgb()
        depth = self.cam.capture_depth()
        position = self.cam.get_position()
        resolution = self.cam.get_resolution()
        extrinsics = self.cam.get_matrix()
        intrinsics = self.cam.get_intrinsic_matrix()
        pointcloud = self.cam.pointcloud_from_depth_and_camera_params(depth,extrinsics,intrinsics)

        obs_dict = dict(
            rgb=rgb,
            depth=depth,
            position=position,
            resolution=resolution,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            pointcloud=pointcloud,
        )
        self.obs.append(obs_dict)

    def save_observations(self, filename:str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.obs, f)



# create observer object
obs = Observer(cam)


# run simulation
start_time = time.time()
while (time.time() - start_time) < SIMULATION_TIME:

    # capture cam
    obs.add_new_observation()

    # step simulation
    pr.step()


obs.save_observations('./outputs/2cam_test_dataset.pkl')


pr.stop()
pr.shutdown()
