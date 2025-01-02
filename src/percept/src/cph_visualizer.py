#!/usr/bin/env python3
import rospy
import numpy as np
import cupoch as cph
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


visualize = 'pcd'


def convert_pointcloud2_to_numpy(msg):
    """
    Convert ROS PointCloud2 message to a NumPy array.
    """
    point_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    return np.array(point_list, dtype=np.float32)

def create_voxel_grid(points, voxel_size=0.05):
    """
    Convert points to a Cupoch VoxelGrid.
    """
    # Create a Cupoch PointCloud
    pointcloud = cph.geometry.PointCloud()
    pointcloud.points = cph.utility.Vector3fVector(points)

    # Create a VoxelGrid from the point cloud
    voxel_grid = cph.geometry.VoxelGrid.create_from_point_cloud(pointcloud, voxel_size)
    return voxel_grid


def main():

    if visualize == 'primitives':
        rospy.init_node('primitive_visualizer', anonymous=True)        
        vis = cph.visualization.Visualizer()
        vis.create_window(window_name="Primitive Visualizer")

        pcd = cph.geometry.PointCloud()
        vis = cph.visualization.Visualizer()
        vis.create_window()

        def primitive_callback(msg):
            points = convert_pointcloud2_to_numpy(msg)

        rospy.Subscriber("/primitives", PointCloud2, primitive_callback)
        # r = rospy.Rate(10)
        initialize = False
        while not rospy.is_shutdown():
            if len(pcd.points) > 0:
                if not initialize:
                    vis.add_geometry(pcd)
                    initialize = True
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            r.sleep()


    if visualize == 'voxel':
        rospy.init_node('voxelgrid_visualizer', anonymous=True)
        vis = cph.visualization.Visualizer()
        vis.create_window(window_name="VoxelGrid Visualizer")

        def voxel_grid_callback(msg, vis):
            
            # Convert PointCloud2 to NumPy array
            points = convert_pointcloud2_to_numpy(msg)

            # Create VoxelGrid from points
            voxel_grid = create_voxel_grid(points)

            vis.update_geometry(voxel_grid)
            vis.poll_events()
            vis.update_renderer()

        # Subscribe to the PointCloud2 topic
        rospy.Subscriber("/voxelgrid", PointCloud2, 
                         voxel_grid_callback, callback_args=vis)
        # ROS spin loop
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            vis.destroy_window()

    if visualize == 'pcd':
        rospy.init_node("pointcloud2_msg", anonymous=True)
        rospy.loginfo("Start pointcloud2_msg node")
        pcd = cph.geometry.PointCloud()
        vis = cph.visualization.Visualizer()
        vis.create_window()

        def pointcloud_callback(data):
            temp = cph.io.create_from_pointcloud2_msg(
                data.data, cph.io.PointCloud2MsgInfo.default(data.width, data.point_step)
            )
            pcd.points = temp.points
            # pcd.colors = temp.colors
            # rospy.loginfo("%d" % len(pcd.points))


        rospy.Subscriber("/pointclouds", PointCloud2, 
                         pointcloud_callback)
        r = rospy.Rate(10)
        initialize = False
        while not rospy.is_shutdown():
            if len(pcd.points) > 0:
                if not initialize:
                    vis.add_geometry(pcd)
                    initialize = True
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            r.sleep()

if __name__ == "__main__":
    main()
