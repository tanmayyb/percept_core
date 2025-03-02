# Perception Pipeline (Percept Core)

## Usage

```bash
ros2 launch percept rs_static.py enable_robot_body_subtraction:=false show_pipeline_delays:=false show_total_pipeline_delay:=false
```
```bash
ros2 run percept fake_panda.py
ros2 run percept fake_realsense.py
```


Pipeline parameters:

- `enable_robot_body_subtraction`: Enable robot body subtraction
- `show_pipeline_delays`: Show pipeline delays
- `show_total_pipeline_delay`: Show total pipeline delay

