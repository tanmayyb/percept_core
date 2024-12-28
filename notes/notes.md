# Notes

####

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
catkin clean --yes
catkin init && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && catkin build
```


Metadata topics:
```

```






#### Realsense Info

```
             /device_0/sensor_0/Depth_0/image/data                              261 msgs    : sensor_msgs/Image  
             /device_0/sensor_0/Depth_0/image/metadata                         4698 msgs    : diagnostic_msgs/KeyValue   
             /device_0/sensor_0/Depth_0/info                                      1 msg     : realsense_msgs/StreamInfo  
             /device_0/sensor_0/Depth_0/info/camera_info                          1 msg     : sensor_msgs/CameraInfo   
             /device_0/sensor_0/Depth_0/tf/0                                      1 msg     : geometry_msgs/Transform  
             /device_0/sensor_0/info                                              1 msg     : diagnostic_msgs/KeyValue   
             /device_0/sensor_0/option/Asic Temperature/description               1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Asic Temperature/value                     1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Depth Units/description                    1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Depth Units/value                          1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Emitter Enabled/description                1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Emitter Enabled/value                      1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Enable Auto Exposure/description           1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Enable Auto Exposure/value                 1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Error Polling Enabled/description          1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Error Polling Enabled/value                1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Exposure/description                       1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Exposure/value                             1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Frames Queue Size/description              1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Frames Queue Size/value                    1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Gain/description                           1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Gain/value                                 1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Inter Cam Sync Mode/description            1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Inter Cam Sync Mode/value                  1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Laser Power/description                    1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Laser Power/value                          1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Output Trigger Enabled/description         1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Output Trigger Enabled/value               1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Projector Temperature/description          1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Projector Temperature/value                1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Stereo Baseline/description                1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Stereo Baseline/value                      1 msg     : std_msgs/Float32   
             /device_0/sensor_0/option/Visual Preset/description                  1 msg     : std_msgs/String    
             /device_0/sensor_0/option/Visual Preset/value                        1 msg     : std_msgs/Float32   
             /device_0/sensor_1/Color_0/image/data                              260 msgs    : sensor_msgs/Image  
             /device_0/sensor_1/Color_0/image/metadata                         5720 msgs    : diagnostic_msgs/KeyValue   
             /device_0/sensor_1/Color_0/info                                      1 msg     : realsense_msgs/StreamInfo  
             /device_0/sensor_1/Color_0/info/camera_info                          1 msg     : sensor_msgs/CameraInfo   
             /device_0/sensor_1/Color_0/tf/0                                      1 msg     : geometry_msgs/Transform  
             /device_0/sensor_1/info                                              1 msg     : diagnostic_msgs/KeyValue   
             /device_0/sensor_1/option/Auto Exposure Priority/description         1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Auto Exposure Priority/value               1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Backlight Compensation/description         1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Backlight Compensation/value               1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Brightness/description                     1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Brightness/value                           1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Contrast/description                       1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Contrast/value                             1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Enable Auto Exposure/description           1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Enable Auto Exposure/value                 1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Enable Auto White Balance/description      1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Enable Auto White Balance/value            1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Exposure/description                       1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Exposure/value                             1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Frames Queue Size/description              1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Frames Queue Size/value                    1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Gain/description                           1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Gain/value                                 1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Gamma/description                          1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Gamma/value                                1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Hue/description                            1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Hue/value                                  1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Power Line Frequency/description           1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Power Line Frequency/value                 1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Saturation/description                     1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Saturation/value                           1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/Sharpness/description                      1 msg     : std_msgs/String    
             /device_0/sensor_1/option/Sharpness/value                            1 msg     : std_msgs/Float32   
             /device_0/sensor_1/option/White Balance/description                  1 msg     : std_msgs/String    
             /device_0/sensor_1/option/White Balance/value                        1 msg     : std_msgs/Float32   
             /device_0/sensor_2/Accel_0/imu/data                                674 msgs    : sensor_msgs/Imu    
             /device_0/sensor_2/Accel_0/imu/metadata                           2022 msgs    : diagnostic_msgs/KeyValue   
             /device_0/sensor_2/Accel_0/imu_intrinsic                             1 msg     : realsense_msgs/ImuIntrinsic
             /device_0/sensor_2/Accel_0/info                                      1 msg     : realsense_msgs/StreamInfo  
             /device_0/sensor_2/Gyro_0/imu/data                                 593 msgs    : sensor_msgs/Imu    
             /device_0/sensor_2/Gyro_0/imu/metadata                            1779 msgs    : diagnostic_msgs/KeyValue   
             /device_0/sensor_2/Gyro_0/imu_intrinsic                              1 msg     : realsense_msgs/ImuIntrinsic

```
