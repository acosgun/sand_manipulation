#!/usr/bin/env python

import os
import sys
import random

import math
import rospy
import rosbag
import cv2
import cv_bridge
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf.transformations as tft
import numpy as np

#from position_control import *
#from kinova_msgs.msg import PoseVelocity
#import kinova_msgs.msg
#import kinova_msgs.srv

from sandman.msg import SandActions
from sandman.msg import PushAction
from sandman.msg import Pixel
import std_msgs.msg
from std_msgs.msg import Int32MultiArray
import std_srvs.srv
import geometry_msgs.msg
import sensor_msgs.msg
from std_msgs.msg import String


#sand_actions_msg = None
image_ref = None

min_rows = 0
max_rows = 480
min_cols = 268
max_cols = 508
tool_size = 30

'''
def sand_actions_callback(msg):
    global sand_actions_msg
    sand_actions_msg = msg
'''
     
def image_capture(msg):
    global image_ref, save_new, diff_img
    global min_rows, max_rows, min_cols, max_cols
    
    cvb = cv_bridge.CvBridge()
    # Convert into opencv matrix
    img = cvb.imgmsg_to_cv2(msg, 'bgr8')
    im_size = img.shape

    if save_new:
        print("Storing Ref Image")
        image_ref = img
        cv2.imwrite(ref_img_name,img)
        save_new = False

    diff_img = cv2.subtract(image_ref, img)
    #accessible robot workspace in image space
    diff_img = diff_img[min_rows:max_rows, min_cols:max_cols]
    # discretisation assuming tool size is 30X30 pixels
    diff_img = cv2.resize(diff_img, (int((max_cols-min_cols)//tool_size),int((max_rows-min_rows)//tool_size)))

if __name__ == '__main__':
    global save_new, diff_img

    # should be the size of robot ws
    diff_img = np.zeros((max_rows-min_rows, max_cols-min_cols, 3), np.uint8)

    bridge = cv_bridge.CvBridge()

    rospy.init_node('texture_detector',anonymous=True) # node name
    
    ref_img_name = rospy.get_param('~ref_texture_img_name', 'refTexture.png')

    save_new = raw_input("Save new image? y/n: ") == 'y'

    if save_new == False:
        image_ref = cv2.imread(ref_img_name)

    image_sub = rospy.Subscriber('/camera/rgb/image_raw', sensor_msgs.msg.Image, image_capture, queue_size=1) #listen robot position
    #actions_sub = rospy.Subscriber('/sand_actions', SandActions, sand_actions_callback, queue_size=1)
    pub = rospy.Publisher('texture', sensor_msgs.msg.Image, queue_size=10)

    # Publish velocity at 100Hz.
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        try:        
            pub.publish(bridge.cv2_to_imgmsg(diff_img, "bgr8"))
        except cv_bridge.CvBridgeError as e:
            print(e)
        r.sleep()
