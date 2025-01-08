# Notes

#### Remap topics:

[Rosbag Roslaunch remap](https://stackoverflow.com/questions/43527081/publish-rosbag-image-on-a-topic-other-than-camera-image-raw)


| ROSBAG Topics: d435i_walking.bag          | col2                  |
| ------------------------------------------- | ----------------------- |
| /device_0/sensor_0/Depth_0/image/data     | /Depth_Image          |
| /device_0/sensor_0/Depth_0/image/metadata | /Depth_Image_Metadata |
| /device_0/sensor_1/Color_0/image/data     | /Color_Image          |
| /device_0/sensor_1/Color_0/image/metadata | /Color_Image_Metadata |
| /device_0/sensor_2/Accel_0/imu/data       | /Accel_Data           |
| /device_0/sensor_2/Accel_0/imu/metadata   | /Accel_Data_Metadata  |
| /device_0/sensor_2/Gyro_0/imu/data        | /Gyro_Data            |
| /device_0/sensor_2/Gyro_0/imu/metadata    | /Gyro_Data_Metadata   |
|                                           |                       |

```
source $PERCEPT_ROOT/devel/setup.bash
roslaunch percept rs_bag_stream.launch 
```

### MISC

#### Build
```
source peract_env/bin/activate
catkin clean --yes
catkin init && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Debug && catkin build
```

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```
