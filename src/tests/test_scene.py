from pyrep import PyRep
import time
from pyrep.objects import VisionSensor, Dummy



pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch('../scenes/scene_reinforcement_learning_env.ttt', headless=False) 
pr.start()  # Start the simulation

# Do some stuff

vs = VisionSensor('my_vision_sensor')  # The name of the sensor in the vrep scene
rgb = vs.capture_rgb()
depth = vs.capture_depth()

start_time = time.time()
while time.time() - start_time < 30.0:
    print('waiting...')
    time.sleep(1.0)


pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application