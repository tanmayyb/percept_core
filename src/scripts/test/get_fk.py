import time
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.objects.object import Object


SIMULATION_TIME = 5.0
HEADLESS    = True
IMAGE_SIZE  = 240
CAMERAS = ['front', 'back']



class ArmWrapper(Arm):
    def __init__(
        self, 
        name: str,
        count: int = 0
    ):
        super().__init__(count, name, 7)    

agent = ArmWrapper('Franka1')


# launch scene
pr = PyRep()
pr.launch(
    './scenes/dualarms_2cam.ttt', 
    headless=HEADLESS,) 
pr.start() 




# run simulation
start_time = time.time()
while (time.time() - start_time) < SIMULATION_TIME:

    # capture obs


    # step simulation
    pr.step()


pr.stop()
pr.shutdown()