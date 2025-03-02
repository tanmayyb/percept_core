# Perception Pipeline (Percept Core)

## Usage

Run Perception Pipeline:
```bash
ros2 launch percept rs_static.py enable_robot_body_subtraction:=false show_pipeline_delays:=false show_total_pipeline_delay:=false
```

Run Fake Hardware:
```bash
ros2 run percept fake_panda.py
ros2 run percept fake_realsense.py
```

Visualize the robot:
```bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(cat assets/robots/franka_panda/panda_ros2.urdf)"
```

Pipeline parameters:

- `enable_robot_body_subtraction`: Enable robot body subtraction
- `show_pipeline_delays`: Show pipeline delays
- `show_total_pipeline_delay`: Show total pipeline delay

