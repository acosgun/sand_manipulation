#!/usr/bin/env python

import os
import sys

import rospy
import rosbag
import cv2
import cv_bridge
import numpy as np
import matplotlib.pyplot as plt
import tf.transformations as tft

import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg
import sensor_msgs.msg
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from sandman.msg import SandActions
from sandman.msg import TapAction
from sandman.msg import PushAction

min_blue = np.array([110,0,0])
max_blue = np.array([255,255,56])

kernel = np.ones((2,2), np.uint8)
num_iter = 2
point_reached = False
data_logged = True
count_against_stuck = 0
img_log_counter = 10000

CURR_POSE = None
centroid = None
vel = None
depth_cam_to_tool = 0.35
im_size = [480, 640]
sand_actions_msg = None
method = None
fontface = cv2.FONT_HERSHEY_SIMPLEX
font_color = (50,50,250)
font_size = 0.7

min_rows = 0
max_rows = 480
min_cols = 268
max_cols = 448
tool_size = 30

desired_z_down = 0.24
desired_z_up = 0.3

goal_point = [240, 320]

tool = "straight" # "straight"

def send_robot_to_home():

    if robot_type == "kinova":
        from position_control import move_to_joint_position
        if tool == "straight":
            #goal_joint_pos = [375.83, 219.43, 67.61, 176.93, 78.07, 27.89, 0.0]
            goal_joint_pos = [383.7, 213.0, 55.1,  181.1,  80.7, 27.1, 0.0]
        else:
            goal_joint_pos = [371.0, 210.0, 50.0, 179.0, 58.0, 43.0, 0.0]
        move_to_joint_position(goal_joint_pos)
        print "1st home reached"

        import time
        time.sleep(0.5)
        #goal_joint_pos = [371.0, 233.0, 70.0, 179.0, 58.0, 38.0, 0.0]
        if tool == "straight":
            goal_joint_pos = [379.8, 235.15, 54.43, 162.54, 112.29, 22.43, 0.0]
        else:
            goal_joint_pos = [371.0, 224.0, 50.0, 179.0, 58.0, 43.0, 0.0]
        move_to_joint_position(goal_joint_pos)
        print "2nd home reached"

        
    #TODO: UR5 implementation


def get_end_points(method):
    end = None
    text = None
    if method == "w":
        end = sand_actions_msg.ann_tap.end
        text = "Neural Net"
    elif method == "x":
        end = sand_actions_msg.polyreg_tap.end
        text = "Poly Regression"
    elif method == "y":
        end = sand_actions_msg.average_tap.end
        text = "Sampled Max Difference"
    elif method == "z":
        end = sand_actions_msg.maxdist_tap.end
        text = "Max Difference"
    return (end, text)

def find_blue(msg):
    global vel, im_size, point_reached
    global centroid, state
    global min_rows, max_rows, min_cols, max_cols, tool_size
    global goal_point
    global data_logged, img_log_counter

    cvb = cv_bridge.CvBridge()
    # Convert into opencv matrix
    img = cvb.imgmsg_to_cv2(msg, 'bgr8')
    im_size = img.shape

    # Threshold the image for blue
    binary_blue = cv2.inRange(img, min_blue, max_blue)
    eroded_blue = cv2.erode(binary_blue, kernel, iterations=num_iter)
    dilated_blue = cv2.dilate(eroded_blue, kernel, iterations=num_iter)

    # Find the contours - saved in blue_contours and red_contours
    im2_b, blue_contours, hierachy_blue = cv2.findContours(dilated_blue.copy(), cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_SIMPLE)
 
    if len(blue_contours) > 0 and blue_contours is not None:
        centroid = detecting_centroids(blue_contours)
        if centroid is not None:
            cv2.circle(img, (centroid[0], centroid[1]), 5, (255,0,0),-1)
    else:
        centroid = None
    
    if state == 0: #Waiting for command
        cv2.putText(img, "Tap Action: Idle", (20, 40), fontface, font_size, font_color, 2)
        pass
    elif state == 1: #VS to tap point
        if not data_logged:            
            ###Data Log
            [end, method_text] = get_end_points(method)
            img_name = "/home/acrv/andrea_sand_data/ros_ws/src/sandman/logs/img" + str(img_log_counter) + ".png"
            file = open("/home/acrv/andrea_sand_data/ros_ws/src/sandman/logs/logged_data.txt", "a")
            file.write(str(img_log_counter) + "\t")
            file.write(str(ord(method)) + "\t")
            file.write(str(end.x) + "\t")
            file.write(str(end.y) + "\t")
            file.write("\n")
            file.close()
            cv2.imwrite(img_name, img)
            img_log_counter = img_log_counter - 1
            data_logged = True
            ###

        if not point_reached:
            print("VS to tapping point")
            [end, method_text] = get_end_points(method)
            #TODO offset for workspaced
            goal_point = [tool_size*end.x +tool_size//2 + min_cols, tool_size*end.y +tool_size//2]
            print(goal_point)
            #goal_point = [320, 240]
            cv2.circle(img, (goal_point[0], goal_point[1]), 6, (0,0,255),-1)
            cv2.putText(img, "Action: tapping point", (20, 40), fontface, font_size, font_color, 2)
            cv2.putText(img, "Method: " + method_text, (20, 70), fontface, font_size, font_color, 2)
            servo_to_point(goal_point, depth_cam_to_tool, img)
        else:
            point_reached = False
            state = 2
    elif state == 2: #go down
        if not point_reached:
            print("go down")
	    #cv2.putText(img, "Action: VS to 2nd point", (20, 40), fontface, font_size, font_color, 2)
            #cv2.putText(img, "Method: " + method_text, (20, 70), fontface, font_size, font_color, 2)
	    cv2.circle(img, (goal_point[0], goal_point[1]), 6, (0,0,255),-1)
            godown(goal_point, depth_cam_to_tool, img)
        else:
            point_reached = False
            state = 3
            print("go down over")
    elif state == 3: #go up
        if not point_reached:
            print("go up")
	    #cv2.putText(img, "Action: VS to 2nd point", (20, 40), fontface, font_size, font_color, 2)
            #cv2.putText(img, "Method: " + method_text, (20, 70), fontface, font_size, font_color, 2)
	    goup()
        else:
            point_reached = False
            state = 4
            print("go up over")
    elif state == 4: #Home
        point_reached = False
        cv2.putText(img, "Action: Sending Robot Home", (20, 40), fontface, font_size, font_color, 2)
        send_robot_to_home()
        state = 0

    #TODO size of low res imqge
    for c in range(0,6):
        for r in range(0,16):
            cv2.circle(img, (tool_size*c+tool_size//2+min_cols, tool_size*r+tool_size//2), 5, (255, 255, 255), 1)

    
    cv2.imshow('TAP Actions', img)
    cv2.moveWindow('TAP Actions', 600, 540)
    cv2.waitKey(1)

def godown(desired_centroid, depth, img):
    global vel, CURR_POSE, point_reached
    global desired_z_down
    global count_against_stuck
    if centroid:
        max_count_against_stuck = 300
        current_pose = CURR_POSE
        position_error_z = desired_z_down - current_pose.position.z 
        K_p = 2 #4
        thresh_z = 0.005	
        cx_b = centroid[0]; 
        cy_b = centroid[1];
        current_centroid = np.array([cx_b, cy_b])      
        thresh_img = 10
        max_norm = 0.05
        e = current_centroid - desired_centroid
        error_norm =np.linalg.norm(e) 

        if ( error_norm > thresh_img or abs(position_error_z) > thresh_z ) and count_against_stuck < max_count_against_stuck:            
            interaction = get_interaction(current_centroid, im_size, depth)
            i_inv = np.linalg.pinv(interaction)
            gain_vs = 0.005 #0.005
            velocity_cam = np.matmul(i_inv, e)
            velocity_b = velocity_cam
            velocity_b[1] = -velocity_b[1]
            velocity_b = gain_vs*velocity_b
            norm = np.linalg.norm(velocity_b)
            if norm > max_norm:
                velocity_b = (velocity_b / norm) * max_norm
            vel_z = K_p * position_error_z
            count_against_stuck += 1
            print("count: ", count_against_stuck)
        else:
            velocity_b = np.array([0, 0])
            vel_z = 0
            point_reached = True
            count_against_stuck = 0
            print("point reached: go down")
    else:
        print("No Assigned Centroid")
        velocity_b = np.array([0, 0])
        vel_z = 0
            
    global robot_type
    if robot_type == 'kinova':
        from kinova_msgs.msg import PoseVelocity
        vel = PoseVelocity()
        vel.twist_linear_x = velocity_b[0]
        vel.twist_linear_y = velocity_b[1]
        vel.twist_linear_z = vel_z
    #TODO: UR5 implementation
    #print("vel_z: ", vel_z)
    #print("position error: ", position_error)

def goup():
    global vel, CURR_POSE, point_reached
    global desired_z_up
    current_pose = CURR_POSE
    position_error = desired_z_up - current_pose.position.z 
    K_p = 4
    thresh = 0.005	
    if abs(position_error) > thresh:            
        vel_z = K_p * position_error
    else:
        point_reached = True
        vel_z = 0
        print("point reached: go up")
    global robot_type
    if robot_type == 'kinova':
        from kinova_msgs.msg import PoseVelocity
        vel = PoseVelocity()
        vel.twist_linear_x = 0
        vel.twist_linear_y = 0
        vel.twist_linear_z = vel_z
    #TODO: UR5 implementation
    #print("vel_z: ", vel_z)
    #print("position error: ", position_error)       

def servo_to_point(desired_centroid, depth, img):
    global vel, point_reached, count_against_stuck
    global im_size, depth_cam_to_tool
    max_count_against_stuck = 300
    if centroid:
        cx_b = centroid[0]; 
        cy_b = centroid[1];
        current_centroid = np.array([cx_b, cy_b])

        vel_z = 0.0
        
        thresh = 10
        max_norm = 0.05
        
        #update desired_centroid for perspective 
        scale_factor = 1.3
        des_centroid_persp = [0.0,0.0]
        des_centroid_persp[0] = int(((desired_centroid[0] - im_size[1]/2) * scale_factor * depth_cam_to_tool / (depth_cam_to_tool - (desired_z_up - desired_z_down))) + im_size[1]/2)
        des_centroid_persp[1] = int(((desired_centroid[1] - im_size[0]/2) * scale_factor * depth_cam_to_tool / (depth_cam_to_tool - (desired_z_up - desired_z_down))) + im_size[0]/2)
       
        if des_centroid_persp[0] > 620:
            des_centroid_persp[0] = 620
        if des_centroid_persp[0] < 20:
            des_centroid_persp[0] = 20 
        if des_centroid_persp[1] > 460:
            des_centroid_persp[1] = 460 
        if des_centroid_persp[1] < 20:
            des_centroid_persp[1] = 20

        e = current_centroid - des_centroid_persp
        error_norm =np.linalg.norm(e) 
        cv2.circle(img, (des_centroid_persp[0], des_centroid_persp[1]), 6, (0,255,255),-1)


        if error_norm > thresh and count_against_stuck < max_count_against_stuck:            
            interaction = get_interaction(current_centroid, im_size, depth)
            i_inv = np.linalg.pinv(interaction)
            gain_vs = 0.008
            velocity_cam = np.matmul(i_inv, e)
            velocity_b = velocity_cam
            velocity_b[1] = -velocity_b[1]

            velocity_b = gain_vs*velocity_b
            norm = np.linalg.norm(velocity_b)
            
            if norm > max_norm:
                velocity_b = (velocity_b / norm) * max_norm

            count_against_stuck += 1
            print("count: ", count_against_stuck)
        else:
            point_reached = True
            velocity_b = np.array([0, 0])
            count_against_stuck = 0
    else:
        print("No Assigned Centroid")
        velocity_b = np.array([0, 0])

    global robot_type
    if robot_type == 'kinova':
        from kinova_msgs.msg import PoseVelocity
        vel = PoseVelocity()
        vel.twist_linear_x = velocity_b[0]
        vel.twist_linear_y = velocity_b[1]
        vel.twist_linear_z = 0
    #TODO: UR5 implementation

    print("vel servo to point: ", velocity_b)

def cam_to_base(velocity):
    # Camera position in base frame
    t= np.array([[-0.51],
                 [-0.05],
                 [0.60 ]])

    transform = np.array([[1, 0,  0,  0,     t[2], -t[1]],
                          [0, -1, 0,  t[2],  0,     t[0]],
                          [0, 0,  -1, -t[1], -t[0], 0   ],
                          [0, 0,  0,  1,     0,     0   ],
                          [0, 0,  0,  0,     -1,    0   ],
                          [0, 0,  0,  0,     0,     -1  ]])

    velocity_b = np.matmul(transform, velocity)

    return velocity_b
    
def get_interaction(centroid, im_size, depth):
    im_width  = im_size[1]
    im_height = im_size[0]
    # Camera to box depth for vs
    Z = depth
    pixel_to_meter = 0.0000036 # Size of pixel in meters
    focal_in_pixels = 430

    # Convert to world frame
    x = (centroid[0] - im_width/2)/focal_in_pixels
    y = (centroid[1] - im_height/2)/focal_in_pixels
 
    L = np.array([[-1/Z, 0],#,    x/Z, x*y,      -(1 + x**2), y],
                  [0,    -1/Z]])#, y/Z, 1 + y**2, -x*y,       -x]])
  
    return L
    
def detecting_centroids(contours):
    list_boxes=[]
    minimum_area=100
    cx=[]
    cy=[]
    # Find the bounding box for each contour
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cx_ = x+(w/2)
        cy_ = y+(h/2)
        area = w*h
        if area > minimum_area:
            list_boxes.append((cx_, cy_, area))

    # Draw centroid of largest blob
    if len(list_boxes) > 0:
        # Sort based on area
        list_boxes.sort(key=lambda x: x[2], reverse=True)
        cx = list_boxes[0][0]
        cy = list_boxes[0][1]
        centroid = (cx, cy)
    else:
        centroid = None

    return centroid


def robot_position_callback(msg):
    # Monitor robot position.
    global CURR_POSE
    CURR_POSE = msg.pose

def force_callback(msg):
    global CURR_FORCE
    CURR_FORCE = msg.wrench.force.z

def sand_actions_callback(msg):
    global sand_actions_msg
    if state == 0:
        sand_actions_msg = msg

def command_generator_callback(msg):
    print "Rcvd Command: " + msg.data

    global state
    global point_reached
    global method
    global data_logged
    
    if msg.data == "w" or msg.data == "x" or msg.data == "y" or msg.data == "z":
        if state == 0:
            method = msg.data
            state = 1
            point_reached = False
            data_logged = False
            print "Enabling Robot Action"

            from position_control import move_to_joint_position
            goal_joint_pos = [383.7, 213.0, 55.1,  181.1,  80.7, 27.1, 0.0]
            move_to_joint_position(goal_joint_pos)
            
    elif msg.data == "h":
        state = 3
        print "Going Home"
        
if __name__ == '__main__':
    rospy.init_node('analisi_img',anonymous=True) # node name

    global robot_type, state
    robot_type = rospy.get_param('~robot_type', 'kinova')
    state = 0
    
    image_sub = rospy.Subscriber('/camera/rgb/image_raw', sensor_msgs.msg.Image, find_blue, queue_size=1)
    actions_sub = rospy.Subscriber('/sand_actions_tap', SandActions, sand_actions_callback, queue_size=1)

    if robot_type == "kinova":
        from kinova_msgs.msg import PoseVelocity
        vel_pub = rospy.Publisher('/m1n6s200_driver/in/cartesian_velocity', PoseVelocity, queue_size=10)
    #TODO: UR5 implementation

    position_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_pose', geometry_msgs.msg.PoseStamped, robot_position_callback, queue_size=1)
    force_sub = rospy.Subscriber('/m1n6s200_driver/out/tool_wrench', geometry_msgs.msg.WrenchStamped, force_callback)

    command_sub = rospy.Subscriber('/commands', std_msgs.msg.String, command_generator_callback)

    #rospy.spin()
    send_robot_to_home()

    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        if vel is not None:
            # print(vel)
            if not point_reached and vel_pub is not None:
                pass
            if state != 0:
                vel_pub.publish(vel)
        r.sleep()
