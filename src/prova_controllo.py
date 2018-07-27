#! /usr/bin/env python

import rospy
import tf.transformations as tft

import numpy as np

import kinova_msgs.msg
import kinova_msgs.srv
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg

import socket
import threading
import time
import zmq

from helpers.transforms import current_robot_pose, publish_tf_quaterion_as_transform, convert_pose, publish_pose_as_transform
from helpers.covariance import generate_cartesian_covariance
from helpers.gripper_action_client import set_finger_positions
from helpers.position_action_client import move_to_position
from helpers.joint_action_client import move_to_joint_position
from std_msgs.msg import String

# variabili globali
hand=-1 # 0: hando ready for a trial | 1: handover started       | 2: handover finish
arm=-1  # 0: arm in start_position   | 1: arm is reaching target | 2:arm is waiting for the handover takes place | 3: arm is going back to the start_position
arrived=-1
velocity_y=0.01
reaching_velocity=[0,0.1,0,0,0,0]
current_velocity=[0,-0,0.0,0,0,2] # [0,-0.15,0,0,0,0] guardando lato oppposto power button X positivo a dx Y positivo uscente Z positovo in alto velocita deg/s
current_position=[0,0,0,0,0,0,0]
current_y=0;
start_position=[-0.144146280527, -0.336400777102, 0.481693327427] # x,y,z (meters) ATTENZIONE: se punto iniziale troppo raccolto il robot si blocca in velocita
start_orientation=[0.675, 0.036, -0.004, 0.737] #quaternion q1,q2,q3,qw
start_joint_position=[ 358.3, 204.7, 47.3, 222.4, 100, 88.0]#358.3, 204.7, 47.3, 222.4, 100, 127.8# [ 358.3, 204.7, 47.3, 222.4, 98.7, 131.0]#[342.2, 175.0, 79.0, 243.0, 84.0, 88.0]# degree [ 347.4, 195.5, 55.6, 230.5, 96.6, 88.3] #[ 354.4, 195.5, 55.6, 230.5, 96.6, 80.4]
initialitation=False #flag per trovare start position
going_back=False

max_reach=0.50 # sbraccio radiale xy assoluto ripsetto alla terna di base

def robot_position_callback(msg): # Monitor tool pose that has within max_reach limit
	global arrived
	global max_reach
	global current_y
	global current_position
	global initialitation
	# Aggiorno la posizione coorrente
	current_position[0]=msg.pose.position.x
	current_position[1]=msg.pose.position.y
	if initialitation:
		current_position[2]=msg.pose.position.z
		current_position[3]=msg.pose.orientation.x
		current_position[4]=msg.pose.orientation.y
		current_position[5]=msg.pose.orientation.z
		current_position[6]=msg.pose.orientation.w

	r=np.sqrt((current_position[0]**2)+(current_position[1]**2)) # calcolo lo sbraccio corrente del robot
	#rospy.loginfo(str(r))
	if r<max_reach:
		arrived=0
		#rospy.loginfo("max distance NOTriched")
	elif r>=max_reach:
		#current_velocity=[0,0,0,0,0,0] # non so se necessario
		arrived=1
		rospy.loginfo("max distance riched")

def hand_state_callback(msg): # Monitor hand state
	global hand
	rospy.loginfo("Handover:"+msg.data)
	data_in=str.split(msg.data,"$")[0] # take the message from the hand without $
	hand=int(data_in)
	#rospy.loginfo("Hand Sate:"+str(hand))
	if hand==1:
		rospy.loginfo("Handover: Started")
	elif hand==2:
		rospy.loginfo("Handover: Finished")

def robot_state_publish(): # Send arm_state to the hand da trasformare in callback per il subscriber da telecamera per calcolare distanza di impatto e triggerare handover
	global arm
	# Construction of the robot message
	msg=str(arm)+"$"
	#publish arm state
	pub.publish(msg)
	pubblica_codifica_arm(arm)

def pubblica_codifica_arm(arm_):
	if arm_==0:
		rospy.loginfo("Arm State published: Start position")
	elif arm_==1:
		rospy.loginfo("Arm State published: Reaching")
	elif arm_==2:
		rospy.loginfo("Arm State published: Waiting for the handover")
	elif arm_==3:
		rospy.loginfo("Arm State published: Coming back Start position")

def go_start_pose():
	global start_orientation
	global start_position
	global start_joint_position
	global arm
	global current_position
	global initialitation

	move_to_joint_position(start_joint_position) # in giunti per essere sicuri che il robot si riporti sempre nella stessa configurazione
	if not initialitation:
		rospy.sleep(1.5)
	elif initialitation:
		rospy.sleep(5) #aspetta di piu per essere sicuri di salvare la posizione giusta
		start_position[0]=current_position[0]
		start_position[1]=current_position[1]
		start_position[2]=current_position[2]
		start_orientation[0]=current_position[3]
		start_orientation[1]=current_position[4]
		start_orientation[2]=current_position[5]
		start_orientation[3]=current_position[6]
		initialitation=False

	arm=0
	rospy.loginfo("start position")

def go_back():
	global start_position
	global start_orientation
	move_to_position(start_position,start_orientation)



def msg_cartesian_velocity(cartVel):
	poseVelCmd = kinova_msgs.msg.PoseVelocity()
	poseVelCmd.twist_linear_x = cartVel[0];
	poseVelCmd.twist_linear_y = cartVel[1];
	poseVelCmd.twist_linear_z = cartVel[2];
	poseVelCmd.twist_angular_x = cartVel[3];
	poseVelCmd.twist_angular_y = cartVel[4];
	poseVelCmd.twist_angular_z = cartVel[5];
	return poseVelCmd

if __name__ == "__main__":
	rospy.init_node('kinova_handover_control',anonymous=True) # node name

	# 1) Definisco publisher and subscriber
	position_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', geometry_msgs.msg.PoseStamped, robot_position_callback, queue_size=1) #listen robot position
	hand_sub=rospy.Subscriber('chatter',String,hand_state_callback,queue_size=1) # listen hand state
	pub=rospy.Publisher('arm',String,queue_size=1)# Publish of the arm state
		# Publish velocity at 100Hz.
	velo_pub = rospy.Publisher('/m1n6s200_driver/in/cartesian_velocity', kinova_msgs.msg.PoseVelocity, queue_size=1)
	r = rospy.Rate(100)

	# 2) Go to the start_position and save it
	initialitation=True
	go_start_pose()
	gp = geometry_msgs.msg.Pose()
	if True:
		gp.orientation.x = start_orientation[0]
		gp.orientation.y = start_orientation[1]
		gp.orientation.z = start_orientation[2]
		gp.orientation.w = start_orientation[3]
		(roll,pitch,yaw) = tft.euler_from_quaternion([gp.orientation.x, gp.orientation.y,gp.orientation.z,gp.orientation.w])

	rospy.loginfo("roll:"+str(roll))
	rospy.loginfo("pitch:"+str(pitch))
	rospy.loginfo("yaw:"+str(yaw))


	# 3) Start del loop di controllo
	count=0
	while not rospy.is_shutdown():
		hand=0

		if (hand==0 and arrived==0):

			arm=1 # arm is reaching
			robot_state_publish()
			velo_pub.publish(kinova_msgs.msg.PoseVelocity(*current_velocity))
		elif hand==2:
			if arm != 0:
				arm=3 # arm is coming back to the start position
				go_back() # to perform a linear trajectory to go back to start position
			else:
				arm=0 #arm is at the start position
			go_start_pose()
			count=0
		else:
			if arm==1: #reaching gia eseguito
				count=count+1; #prima di abbassare la soglia della cella aspetta 500ms (5 cicli)
				if count > 200:
					arm=2 # arm is waiting for the handover
			elif arm==0:
				arm=0 #arm is waiting for the trial

		#publish arm state
		#robot_state_publish()
		#publish arm state
		#robot_state_publish()
		r.sleep()
