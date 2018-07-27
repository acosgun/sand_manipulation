# Borrowed and modified from the kinova-ros examples.

import rospy
import actionlib
import kinova_msgs.msg
from kinova_msgs.msg import JointAngles
import geometry_msgs.msg
import std_msgs.msg

def move_to_position(position, orientation):
    """Send a cartesian goal to the action server."""
    global position_client
    if position_client is None:
        init()

    goal = kinova_msgs.msg.ArmPoseGoal()
    goal.pose.header = std_msgs.msg.Header(frame_id=('m1n6s200_link_base'))
    goal.pose.pose.position = geometry_msgs.msg.Point(
        x=position[0], y=position[1], z=position[2])
    goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
        x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

    position_client.send_goal(goal)

    if position_client.wait_for_result(rospy.Duration(10.0)):
        return position_client.get_result()
    else:
        position_client.cancel_all_goals()
        print('        the cartesian action timed-out')
        return None


def move_to_joint_position(angle_set):

    joint_action_address = '/m1n6s200_driver/joints_action/joint_angles'
    
    joint_position_client = actionlib.SimpleActionClient(joint_action_address, kinova_msgs.msg.ArmJointAnglesAction)
    joint_position_client.wait_for_server()

    import numpy as np
    goal = kinova_msgs.msg.ArmJointAnglesGoal()
    goal.angles.joint1 = angle_set[0]
    goal.angles.joint2 = angle_set[1]
    goal.angles.joint3 = angle_set[2]
    goal.angles.joint4 = angle_set[3]
    goal.angles.joint5 = angle_set[4]
    goal.angles.joint6 = angle_set[5]
    goal.angles.joint7 = angle_set[6]

    joint_position_client.send_goal(goal)

    if joint_position_client.wait_for_result(rospy.Duration(20.0)):
        return joint_position_client.get_result()
    else:
        print('        the joint angle action timed-out')
        joint_position_client.cancel_all_goals()
        return None
    
action_address = '/m1n6s200_driver/pose_action/tool_pose'
position_client = None
joint_position_client = None
        
def init():
    global position_client    
    position_client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmPoseAction)
    position_client.wait_for_server()

