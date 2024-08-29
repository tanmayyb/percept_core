import os, time, pickle
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.objects import VisionSensor

SCENE = './scenes/dualarms_3cam.ttt'
DATASET_SAVE_FILEPATH = './outputs/dualarms_3cam/dualarms_3cam_with_fk.pkl'
SIMULATION_TIME = 5.0
HEADLESS    = True
IMAGE_SIZE  = 240
ARM_NAME = 'panda'
CAMERAS = ['cam1', 'cam2', 'cam3']


# setup and launch scene
pr = PyRep()
pr.launch(
    SCENE, 
    headless=HEADLESS,) 
pr.start() 


# wrapper for agents
class ArmWrapper(Arm):
    def __init__(
        self, 
        name: str,
        count: int = 0
    ):
        super().__init__(count, name, 7)  
        self.name = f'{name}{count}'

# setup agents
# from Pyrep's robots/arms/arm.py
# suffix = '' if count == 0 else '#%d' % (count - 1) 
agent1 = ArmWrapper(ARM_NAME,)
agent2 = ArmWrapper(ARM_NAME, count=1) 

agent_dict = dict(arms=[
    agent1, 
    agent2
])


class Observer():
    def __init__(
        self,
        CAMERA_NAMES:list,
        IMAGE_SIZE:int,
        AGENTS:dict,
    ):
        # setup cameras
        self.CAMERA_NAMES   = CAMERA_NAMES
        self.cameras    = dict()
        self.IMAGE_SIZE = IMAGE_SIZE

        # setup cameras and set resolution
        for cam_name in CAMERAS:
            cam = VisionSensor(cam_name)  
            cam.set_resolution([IMAGE_SIZE,IMAGE_SIZE])
            self.cameras[cam_name] = cam

        # list for storing observations
        self.observations = list()

        # setup agents
        self.agents = AGENTS

    def store_new_observation(self) -> None:
        obs = dict()
        
        # store cam obs
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = self.get_camera_observation(cam) 

        # store arm joints obs
        for agent in self.agents['arms']:
            obs[agent.name] = self.get_agent_observation(agent)

        # store observation
        self.observations.append(obs)

    def get_agent_observation(self, agent) -> dict:
        joint_pos = agent.get_joint_positions()
        global_pos = agent.get_position()
        global_ang = agent.get_orientation()
        return dict( 
            joint_pos = joint_pos,
            global_pos = global_pos,
            global_ang = global_ang,
        )

    def get_camera_observation(self, cam:VisionSensor) -> dict:
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

    def save_observations(self, filepath:str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.observations, f)


# setup observer
observer_handle = Observer(
    CAMERAS,
    IMAGE_SIZE,
    agent_dict,
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