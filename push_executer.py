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
from sandman.msg import PushAction

min_blue = np.array([110,0,0])
max_blue = np.array([255,255,56])

kernel = np.ones((2,2), np.uint8)
num_iter = 2
point_reached = False
data_logged = True
count_against_stuck = 0

CURR_POSE = None
centroid = None
vel = None
depth_cam_to_tool = 0.4
im_size = [480, 640]
sand_actions_msg = None
method = None
fontface = cv2.FONT_HERSHEY_SIMPLEX
font_color = (50,50,250)
font_size = 0.7
img_log_counter = 0


tool = "straight" # "straight"

def send_robot_to_home():

    if robot_type == "kinova":
        from position_control import move_to_joint_position
        #goal_joint_pos = [371.0, 220.0, 70.0, 179.0, 58.0, 38.0, 0.0]
        if tool == "straight":
            goal_joint_pos =  [379.8, 220.15, 54.43, 162.54, 112.29, 22.43, 0.0]
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


def get_start_end_points(method):
    start = None
    end = None
    text = None
    if method == "a":
        start = sand_actions_msg.ann_push.start
        end = sand_actions_msg.ann_push.end
        text = "Neural Net"
        #print("VS to A: Executing ANN")
    elif method == "b":
        start = sand_actions_msg.polyreg_push.start
        end = sand_actions_msg.polyreg_push.end
        text = "Poly Regression"
        #print("VS to A: Executing Poly Regression")
    elif method == "c":
        start = sand_actions_msg.average_push.start
        end = sand_actions_msg.average_push.end
        text = "Avg Contour"
        #print("VS to A: Executing Average Contour")
    elif method == "d":
        start = sand_actions_msg.maxdist_push.start
        end = sand_actions_msg.maxdist_push.end
        text = "Max Contour"
        #print("VS to A: Executing Max Contour Distance")
    return (start, end, text)

def find_blue(msg):
    global vel, im_size, point_reached
    global centroid, state
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

    enable_force = False
    
    if state == 0: #Waiting for command
        cv2.putText(img, "Action: Idle", (20, 40), fontface, font_size, font_color, 2)
        pass
    elif state == 1: #VS to A        
        if not data_logged:
            [start, end, method_text] = get_start_end_points(method)
            img_name = "/home/acrv/andrea_sand_data/ros_ws/src/sandman/logs/img" + str(img_log_counter) + ".png"
            img_log_counter = img_log_counter + 1

            file = open("/home/acrv/andrea_sand_data/ros_ws/src/sandman/logs/logged_data.txt", "a")
            file.write(str(img_log_counter) + "\t")
            file.write(str(ord(method)) + "\t")
            file.write(str(start.x) + "\t")
            file.write(str(start.y) + "\t")
            file.write(str(end.x) + "\t")
            file.write(str(end.y) + "\t")
            global sand_actions_msg
            for i in xrange(len(sand_actions_msg.contour)):
                file.write(str(sand_actions_msg.contour[i]) + "\t")
            file.write("\n")
            file.close()
            cv2.imwrite(img_name, img)
            
            data_logged = True

        if not point_reached:
            print("VS to A")

            [start, end, method_text] = get_start_end_points(method)
            
            #print "Start: " + str(start.x) + ", " + str(start.y)
            #print "End: " + str(end.x) + ", " + str(end.y)
            #goal_x = points[0]
            #goal_y = points[1]
            #goal_point = [goal_x, goal_y]
            
            goal_point = [start.x, start.y]

            #cv2.circle(img, (goal_point[0], goal_point[1]), 6, (0,0,255),-1)
            cv2.arrowedLine(img, (start.x, start.y), (end.x, end.y), font_color, 4)
            cv2.putText(img, "Action: VS to 1st point", (20, 40), fontface, font_size, font_color, 2)                        
            cv2.putText(img, "Method: " + method_text, (20, 60), fontface, font_size, font_color, 2)
            
            servo_to_point(goal_point, depth_cam_to_tool, enable_force)
        else:
            point_reached = False
            state = 2
    elif state == 2: #VS to B
        if not point_reached:
            print("VS to B")
            
            [start, end, method_text] = get_start_end_points(method)
            
            #goal_x = points[2]
            #goal_y = points[3]
            #goal_point = [goal_x, goal_y]

            goal_point = [end.x, end.y]
            
            #cv2.circle(img, (goal_point[0], goal_point[1]), 5, (0,0,255),-1)
            cv2.arrowedLine(img, (start.x, start.y), (end.x, end.y), font_color, 4)
            cv2.putText(img, "Action: VS to 2nd point", (20, 40), fontface, font_size, font_color, 2)
            cv2.putText(img, "Method: " + method_text, (20, 70), fontface, font_size, font_color, 2)
            
            servo_to_point(goal_point, depth_cam_to_tool, enable_force)
        else:
            point_reached = False
            state = 3
    elif state == 3: #Home
        if not point_reached:
            print("Going up!")
            goup()
        else:
            point_reached = False
            state = 4
            print("Gone up")
    elif state == 4:
        point_reached = False
        cv2.putText(img, "Action: Sending Robot Home", (20, 40), fontface, font_size, font_color, 2)
        send_robot_to_home()
        state = 0

    cv2.imshow('Executed Actions', img)
    cv2.moveWindow('Executed Actions', 1280, 583)
    cv2.waitKey(1)


def goup():
    global vel, CURR_POSE, point_reached
    desired_z_up = 0.3
    current_pose = CURR_POSE
    position_error = desired_z_up - current_pose.position.z 
    K_p = 4
    thresh = 0.005	
    if abs(position_error) > thresh:            
        vel_z = K_p * position_error
    else:
        point_reached = True
        vel_z = 0.0
        print("point reached: go up")
    global robot_type
    if robot_type == 'kinova':
        from kinova_msgs.msg import PoseVelocity
        vel = PoseVelocity()
        vel.twist_linear_x = 0.0
        vel.twist_linear_y = 0.0
        vel.twist_linear_z = vel_z
    #TODO: UR5 implementation
    #print("vel_z: ", vel_z)
    #print("position error: ", position_error)   


def servo_to_point(desired_centroid, depth, enable_z):
    global vel, point_reached, CURR_FORCE, CURR_POSE, count_against_stuck
    max_count_against_stuck = 300
    if centroid:
        cx_b = centroid[0]; cy_b = centroid[1];
        current_centroid = np.array([cx_b, cy_b])

        vel_z = 0.0
        
        thresh = 10
        max_norm = 0.05
        e = current_centroid - desired_centroid
        error_norm =np.linalg.norm(e) 

        if error_norm > thresh and count_against_stuck < max_count_against_stuck:            
            interaction = get_interaction(current_centroid, im_size, depth)
            i_inv = np.linalg.pinv(interaction)

            gain_vs = 0.008
            velocity_cam = np.matmul(i_inv, e)
            # velocity_b = cam_to_base(velocity_cam)[:2]
            velocity_b = velocity_cam
            velocity_b[1] = -velocity_b[1]

            velocity_b = gain_vs*velocity_b
            norm = np.linalg.norm(velocity_b)
            
            if norm > max_norm:
                velocity_b = (velocity_b / norm) * max_norm

            if enable_z:
                # Contact Management
                #desired_force = -14.0
                desired_force = -5.0
                force_error = desired_force - CURR_FORCE
                #K_d = 0.008
                K_d = 0.0 #0.001
                   
                current_pose = CURR_POSE

                if tool == "straight":
                    desired_z = 0.19
                else:
                    desired_z = 0.15

                position_error = desired_z - current_pose.position.z 
                #print("current z position: ", current_pose.position.z)
                K_p = 5

                #vel_z = K_d * force_error
                vel_z = K_d * force_error + K_p * position_error
                #vel_z = K_p * position_error

                max_vel_z = 0.03                    
                #vel_z = np.clip(vel_z, -max_vel_z, max_vel_z)  
                #print(vel_z)   

            count_against_stuck += 1
            print("count: ", count_against_stuck)
               
        else:
            point_reached = True
            #print("Point Reached?", point_reached)
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
        vel.twist_linear_z = vel_z
    #TODO: UR5 implementation

    print("vel: ", velocity_b)

def goto_relative_height(height_delta):
    global point_reached
    goal = CURR_POSE
    pos = [goal.position.x, goal.position.y, goal.position.z + height_delta]
    ori = [goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w]
    #ori = [-0.13, -0.69, 0.277, 0.655]

    if robot_type == "kinova":
        from position_control import move_to_position
        move_to_position(pos, ori)
    #TODO: UR5 implementation
        
    point_reached = False

def cam_to_base(velocity):
    # Camera position in base frame
    t= np.array([[-0.51],
                 [-0.05],
                 [0.45 ]])

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

    # L = np.array([[-1/Z, 0,    x/Z, x*y,      -(1 + x**2), y],
    #               [0,    -1/Z, y/Z, 1 + y**2, -x*y,       -x]])
 
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
    
    if msg.data == "a" or msg.data == "b" or msg.data == "c" or msg.data == "d":
        if state == 0:
            method = msg.data
            state = 1
            point_reached = False
            data_logged = False
            print "Enabling Robot Action"
    elif msg.data == "h":
        state = 3
        print "Going Home"
        
if __name__ == '__main__':
    rospy.init_node('analisi_img',anonymous=True) # node name

    global robot_type, state
    robot_type = rospy.get_param('~robot_type', 'kinova')
    state = 0
    
    image_sub = rospy.Subscriber('/camera/rgb/image_raw', sensor_msgs.msg.Image, find_blue, queue_size=1)
    actions_sub = rospy.Subscriber('/sand_actions', SandActions, sand_actions_callback, queue_size=1)

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
                vel_pub.publish(vel)
        r.sleep()
