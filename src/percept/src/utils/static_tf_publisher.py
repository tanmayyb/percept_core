#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from camera_helpers import create_tf_msg_from_xyzrpy

def load_static_tf_frames(): # using config file

    # load config files
    static_camera_config = rospy.get_param("static_camera_config/", None)  
    scene_config = rospy.get_param("scene_config/", None)  
    agent_config = rospy.get_param("agent_pos/", None)

    # create tf messages
    static_tf_frames = list()
    
    # camera frames
    camera_frames = list()
    for camera_name, camera_config in static_camera_config.items():
        position = camera_config['pose']['position']
        orientation = camera_config['pose']['orientation']
        child_frame = f'{camera_name}_link'
        msg = create_tf_msg_from_xyzrpy(child_frame, 
            position['x'], position['y'], position['z'],
            orientation['roll'], orientation['pitch'], orientation['yaw']
        )
        camera_frames.append(msg)

    # agent frame
    agent_frames=list()
    agent_frame = 'agent_frame'
    msg = create_tf_msg_from_xyzrpy(agent_frame, 
        agent_config['x'], agent_config['y'], agent_config['z'],
        agent_config['roll'], agent_config['pitch'], agent_config['yaw']
    )
    agent_frames.append(msg)

    # combine all frames
    static_tf_frames = camera_frames + agent_frames

    return static_tf_frames


def main():
    rospy.init_node('static_tf_publisher')

    # Load transforms
    static_tf_frames = load_static_tf_frames()
    if not static_tf_frames:
        rospy.logerr("No valid static transforms found. Exiting.")
        return

    # Publish static transforms
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    broadcaster.sendTransform(static_tf_frames)

    rospy.loginfo("Static transforms published successfully.")

    # spin once
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
