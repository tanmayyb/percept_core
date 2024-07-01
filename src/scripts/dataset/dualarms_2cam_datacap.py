import os, sys, logging, time, pickle
import numpy as np
from pyrep import PyRep
from pyrep.objects import VisionSensor, Dummy


SIMULATION_TIME = 5.0
HEADLESS    = True
IMAGE_SIZE  = 240
CAMERAS = ['front', 'back']
DATASET_SAVE_FILEPATH = './outputs/dualarms_2cam/dualarms_2cam.pkl'

class Observer():
    def __init__(
        self,
        CAMERA_NAMES:list,
        IMAGE_SIZE:int,
    ):
        # store main vars
        self.CAMERA_NAMES   = CAMERA_NAMES
        self.cameras    = dict()
        self.IMAGE_SIZE = IMAGE_SIZE

        # list for storing observations
        self.observations = list()

        # setup cameras and set resolution
        for cam_name in CAMERAS:
            cam = VisionSensor(cam_name)  
            cam.set_resolution([IMAGE_SIZE,IMAGE_SIZE])
            self.cameras[cam_name] = cam

    def store_new_observation(self) -> None:
        obs = dict()
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = self.get_observation(cam) 
        self.observations.append(obs)

    def get_observation(self, cam:VisionSensor) -> dict:
        rgb = cam.capture_rgb()
        depth = cam.capture_depth()
        position = cam.get_position()
        resolution = cam.get_resolution()
        extrinsics = cam.get_matrix()
        intrinsics = cam.get_intrinsic_matrix()
        pointcloud = cam.capture_pointcloud()

        return dict(
            rgb=rgb,
            depth=depth,
            position=position,
            resolution=resolution,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            pointcloud=pointcloud)

    def save_observations(self, filename:str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.observations, f)



# launch scene
pr = PyRep()
pr.launch(
    './scenes/dualarms_2cam.ttt', 
    headless=HEADLESS,) 
pr.start() 




observer_handle = Observer(
    CAMERAS,
    IMAGE_SIZE
)


# run simulation
start_time = time.time()
while (time.time() - start_time) < SIMULATION_TIME:

    # capture obs
    observer_handle.store_new_observation()

    # step simulation
    pr.step()

# save dataset
observer_handle.save_observations(DATASET_SAVE_FILEPATH)

pr.stop()
pr.shutdown()
