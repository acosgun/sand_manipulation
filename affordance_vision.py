#!/usr/bin/env python

import os
import sys

import rospy
import rosbag
import cv2
import cv_bridge
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf.transformations as tft
from position_control import *

from kinova_msgs.msg import PoseVelocity
from kinova_msgs.msg import JointAngles
import kinova_msgs.msg
import kinova_msgs.srv
import std_msgs.msg
import std_srvs.srv
import geometry_msgs.msg
import sensor_msgs.msg
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray

kernel = np.ones((2,2), np.uint8)
num_iter = 2
point_reached = False

CURR_POSE = None
centroid = None
vel = None
depth_cam_to_tool = 0.4
im_size = [480, 640]
points = None
JOINTS = None

def move_to_pos(pos):
    while np.linalg.norm(np.asarray(JOINTS) - np.asarray(pos)) > 1:
        move_to_joint_position(pos)
    print "Position Reached"
    import time
    time.sleep(0.5)

def find_color(img, min_colors, max_colors):
    global vel, im_size, point_reached
    global centroid, state

    # Threshold the image for blue
    binary  = cv2.inRange(img, min_colors, max_colors)
    eroded  = cv2.erode(binary, kernel, iterations=num_iter)
    dilated = cv2.dilate(eroded, kernel, iterations=num_iter)
    cv2.imshow("Thresh", dilated)

    # Find the contours - saved in blue_contours and red_contours
    im2, contours, hierachy= cv2.findContours(dilated.copy(), cv2.RETR_TREE,
                                                         cv2.CHAIN_APPROX_SIMPLE)
 
    boxes = []
    if len(contours) > 0 and contours is not None:
        boxes = detecting_centroids(contours)

    return boxes


def vision(msg):
    global vel, im_size, point_reached
    global centroid, state

    min_blue = np.array([30,0,0])
    max_blue = np.array([255,80,30])

    min_red = np.array([0,0,20])
    max_red = np.array([30,80,255])

    cvb = cv_bridge.CvBridge()
    # Convert into opencv matrix
    img = cvb.imgmsg_to_cv2(msg, 'bgr8')
    im_size = img.shape

    blue_boxes = find_color(img, min_blue, max_blue)
    red_boxes = find_color(img, min_red, max_red)
    for box in blue_boxes:
        cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,0,0), 3)
    for box in red_boxes:
        cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,0,255), 3)
    cv2.imshow("Blue", img)
    cv2.waitKey(1)


def detecting_centroids(contours):
    # list_boxes=[]
    boxes = []
    minimum_area=8000
    cx=[]
    cy=[]
    # Find the bounding box for each contour
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # cx_ = x+(w/2)
        # cy_ = y+(h/2)
        area = w*h
        if area > minimum_area:
            boxes.append([x,y,w,h])
            # list_boxes.append((cx_, cy_, area))

    # # Draw centroid of largest blob
    # if len(list_boxes) > 0:
    #     # Sort based on area
    #     list_boxes.sort(key=lambda x: x[2], reverse=True)
    #     cx = list_boxes[0][0]
    #     cy = list_boxes[0][1]
    #     centroid = (cx, cy)
    # else:
    #     centroid = None

    return boxes

def robot_position_callback(msg):
    # Monitor robot position.
    global JOINTS
    JOINTS = [msg.joint1, msg.joint2, msg.joint3, msg.joint4, msg.joint5, msg.joint6, msg.joint7]
    # print(JOINTS)

def reset():
    home = [241.6, 165.8, 52.665, 385.23, 392.93, -239.86, 0.0]
    move_to_pos(home)

if __name__ == '__main__':
    rospy.init_node('analisi_img',anonymous=True) # node name

    global state
    state = 0
    
    image_sub = rospy.Subscriber('/camera/rgb/image_raw', sensor_msgs.msg.Image, vision, queue_size=1) #listen robot position

    # pub=rospy.Publisher('camera',String,queue_size=1)# Publish of the image information

    vel_pub = rospy.Publisher('/m1n6s200_driver/in/cartesian_velocity', PoseVelocity, queue_size=10)
    position_sub = rospy.Subscriber('/m1n6s200_driver/out/joint_angles', JointAngles, robot_position_callback, queue_size=1)
    #rospy.spin()

    r = rospy.Rate(100)

    t = 0
 
    while not rospy.is_shutdown():
        
        if JOINTS is not None:
            if t == 0:
                reset()
            t += 1

    
        if vel is not None:
            print(vel)
        r.sleep()
