import time
from pyrep import PyRep
from pyrep.robots.arms.arm import Arm
from pyrep.objects.object import Object


SIMULATION_TIME = 5.0
HEADLESS    = True

pr = PyRep()
pr.launch(
    './scenes/dualarms_2cam.ttt', 
    headless=HEADLESS,) 
pr.start() 


# run simulation
start_time = time.time()
while (time.time() - start_time) < SIMULATION_TIME:

    # do something
    # ...

    # step simulation
    pr.step()


pr.stop()
pr.shutdown()