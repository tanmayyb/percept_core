


Stream ROSBAG:
```
rosbag play d435i_walking.bag -l --topics \
/device_0/sensor_1/Color_0/image/data \
/device_0/sensor_0/Depth_0/image/data
```

Remap topics:

[Publish ROSBAG image on a topic other than camera_image_raw](https://stackoverflow.com/questions/43527081/publish-rosbag-image-on-a-topic-other-than-camera-image-raw)
```

rosbag play d435i_walking.bag -l \
--topics \
/device_0/sensor_1/Color_0/image/data \
/device_0/sensor_0/Depth_0/image/data \
--remap \
/device_0/sensor_1/Color_0/image/data:=/colorImage \
/device_0/sensor_0/Depth_0/image/data:=/depthImage

```