

#### RVIZ
```
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 -1.5 map camera_1_link
```

#### Observations
```
 self.observations = [
     {
         # Camera data for each camera (cam1, cam2, cam3)
         "cam1": {
             "rgb": array(...),        # RGB image data
             "depth": array(...),      # Depth image data
             "position": array(...),   # Camera position
             "resolution": array(...), # Camera resolution
             "extrinsics": array(...), # Camera extrinsic matrix
             "intrinsics": array(...), # Camera intrinsic matrix
             "pointcloud": array(...), # Point cloud data
         },
         "cam2": { ... },
         "cam3": { ... },

         # Robot arm data for each arm (panda0, panda1)
         "panda0": {
             "joint_pos": array(...),  # Joint positions
             "global_pos": array(...), # Global position
             "global_ang": array(...), # Global orientation
         },
         "panda1": { ... }
     },
     # Next timestep...
 ]
```



#### Realsense Info

`/device_0/sensor_0/Depth_0/info/camera_info`
```
header: 
  seq: 0
  stamp: 
    secs: 0
    nsecs:         0
  frame_id: ''
height: 720
width: 1280
distortion_model: "Brown Conrady"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [637.8672485351562, 0.0, 637.4591674804688, 0.0, 637.8672485351562, 361.9825439453125, 0.0, 0.0, 1.0]
R: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
P: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---
```


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


```
	<arg 	name = "selectedTopics"  
				default = "--topics \
				/device_0/sensor_0/Depth_0/image/data \
				/device_0/sensor_0/Depth_0/image/metadata \
				/device_0/sensor_1/Color_0/image/data \
				/device_0/sensor_1/Color_0/image/metadata \
				/device_0/sensor_2/Accel_0/imu/data \
				/device_0/sensor_2/Accel_0/imu/metadata \
				/device_0/sensor_2/Gyro_0/imu/data \
				/device_0/sensor_2/Gyro_0/imu/metadata" />

	<remap from = "/device_0/sensor_0/Depth_0/image/data" to = "/Depth_Image" />
	<remap from = "/device_0/sensor_0/Depth_0/image/metadata" to = "/Depth_Image_Metadata" />
	<remap from = "/device_0/sensor_1/Color_0/image/data" to = "/Color_Image" />
	<remap from = "/device_0/sensor_1/Color_0/image/metadata" to = "/Color_Image_Metadata" />
	<remap from = "/device_0/sensor_2/Accel_0/imu/data" to = "/Accel_Data" />
	<remap from = "/device_0/sensor_2/Accel_0/imu/metadata" to = "/Accel_Data_Metadata" />
	<remap from = "/device_0/sensor_2/Gyro_0/imu/data" to = "/Gyro_Data" />
	<remap from = "/device_0/sensor_2/Gyro_0/imu/metadata" to = "/Gyro_Data_Metadata" />	
```

