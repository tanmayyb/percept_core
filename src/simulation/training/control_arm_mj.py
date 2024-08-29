import mujoco
import mujoco.viewer as viewer
import numpy as np

# Load the MuJoCo model
model_path = "./src/models/panda.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Viewer setup
v = viewer.launch_passive(model, data)

# Define a target joint position for the arm (7 joints for the Panda arm)
target_position = np.array([0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])

# Proportional gain for the controller
kp = 50.0

# Simulation loop
while v.is_running():
    # Calculate the difference between the current and target joint positions
    joint_error = target_position - data.qpos[:7]

    # Apply a simple proportional controller to the joints
    control_signal = kp * joint_error

    # Apply the control signal to the actuators
    data.ctrl[:7] = control_signal

    # Step the simulation
    mujoco.mj_step(model, data)

    # Render the simulation in the viewer
    v.sync()

# Properly close the viewer when the loop ends
v.close()