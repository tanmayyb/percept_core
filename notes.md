Stream ROSBAG:

```
rosbag play d435i_walking.bag -l --topics \
/device_0/sensor_1/Color_0/image/data \
/device_0/sensor_0/Depth_0/image/data
```

Remap topics:

[Rosbag Roslaunch remap](https://stackoverflow.com/questions/43527081/publish-rosbag-image-on-a-topic-other-than-camera-image-raw)


| ROSBAG Topics: d435i_walking.bag          | col2                  |
| ------------------------------------------- | ----------------------- |
| /device_0/sensor_0/Depth_0/image/data     | /device_0/Depth_Image |
| /device_0/sensor_0/Depth_0/image/metadata |                       |
| /device_0/sensor_1/Color_0/image/data     | /device_0/Color_Image |
| /device_0/sensor_1/Color_0/image/metadata |                       |
| /device_0/sensor_2/Accel_0/imu/data       |                       |
| /device_0/sensor_2/Accel_0/imu/metadata   |                       |
| /device_0/sensor_2/Gyro_0/imu/data        |                       |
| /device_0/sensor_2/Gyro_0/imu/metadata    |                       |
|                                           |                       |
