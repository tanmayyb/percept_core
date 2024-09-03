import mujoco
import mujoco.viewer as viewer
import numpy as np
import pygame
import sys
import time

# Initialize Pygame and the joystick
pygame.init()
pygame.joystick.init()

# Ensure a joystick is connected
if pygame.joystick.get_count() == 0:
    print("No joystick connected")
    sys.exit()



# Get the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Print the number of axes
num_axes = joystick.get_numaxes()
print(f"Number of axes: {num_axes}")


# Load the MuJoCo model
model_path = "./src/simulation/assets/franka_panda/panda.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Viewer setup
v = viewer.launch_passive(model, data)

# Define a target joint position for the arm (7 joints for the Panda arm)
target_position = np.array([0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])

# scaling_factor = 1.0  # Scale joystick input to joint positions
scaling_factor = np.pi/2.0


# Proportional gain for the controller
# kp = 50.0

# PID parameters for each joint (these need tuning)
Kp = np.array([0.001]*7)  # Proportional gain
Ki = np.array([0.005]*7)  # Integral gain
Kd = np.array([0.001]*7)  # Derivative gain

# Initialize PID control variables
integral = np.zeros(7)
previous_error = np.zeros(7)
# dt = 0.01  # Time step (adjust based on your simulation rate)



last = time.time()

# Simulation loop
while v.is_running():
    # Handle events
    pygame.event.pump()

    # Get joystick axes (assuming the first 7 axes control the 7 joints)
    joystick_values = np.array([joystick.get_axis(i) for i in range(6)]+[0.])

    # Scale joystick input to desired joint positions or velocities
    target_position = scaling_factor * joystick_values  # Adjust scaling as needed
    print(target_position)

    dt = time.time()-last
    # Calculate the difference between the current and target joint positions
    error = target_position - data.qpos[:7]
    integral += error * dt
    derivative = (error - previous_error) / (dt)
    control_signal = Kp * error + Ki * integral + Kd * derivative
    
    # Apply the control signal to the actuators
    data.ctrl[:7] = control_signal

    # Step the simulation
    mujoco.mj_step(model, data)

    # Render the simulation in the viewer
    v.sync()
    last=time.time()
    

# Properly close the viewer when the loop ends
v.close()
pygame.quit()