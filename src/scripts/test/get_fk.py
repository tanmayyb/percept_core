import time, pickle
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.objects import VisionSensor

# SCENE = './scenes/rlbench_2cam.ttt'
SCENE = './scenes/dualarms_2cam.ttt'
SIMULATION_TIME = 5.0
HEADLESS    = True
IMAGE_SIZE  = 240
CAMERAS = ['front', 'back']
DATASET_SAVE_FILEPATH = './outputs/dualarms_2cam/dualarms_2cam_with_joints.pkl'



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

# setup agent
agent1 = ArmWrapper('Panda')
agent2 = ArmWrapper('Panda', count=1)

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
        obs = dict(cams=dict(), joints=dict())
        
        # store cam obs
        for cam_name, cam in self.cameras.items():
            obs['cams'][cam_name] = self.get_camera_observation(cam) 

        # store arm joints obs
        for agent in self.agents['arms']:
            obs['joints'][agent._handle] = agent.get_joint_positions()

        self.observations.append(obs)

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

    def save_observations(self, filename:str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.observations, f)


# setup observer
agent_dict = dict(arms=[agent1, agent2])
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