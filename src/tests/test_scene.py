import gc
import logging
import os
import sys
import time


from pyrep import PyRep
from pyrep.objects import VisionSensor, Dummy


pr = PyRep()
pr.launch(
    './scenes/test_scene_reinforcement_learning_env.ttt', 
    headless=False
) 
pr.start() 



front_cam = VisionSensor('front_rgbd')  # The name of the sensor in the vrep scene


start_time = time.time()


while (time.time() - start_time) < 30.0:
    rgb_obs = front_cam.capture_rgb()
    depth_obs = front_cam.capture_depth()
    
    print(rgb_obs)
    print(depth_obs)

    pr.step()


pr.stop()
pr.shutdown()