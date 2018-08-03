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
max_cols = 448
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

    diff_img_raw = cv2.subtract(image_ref, img)

    enable_masking = True

    if enable_masking:
        img_mod = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        image_ref_mod = cv2.cvtColor(image_ref.copy(), cv2.COLOR_BGR2GRAY)
        
        kernel = np.ones((3,3), np.uint8)
        
        img_mod = cv2.dilate(img_mod, kernel, iterations=1)
        image_ref_mod = cv2.dilate(image_ref_mod, kernel, iterations=1)
        img_mod = cv2.erode(img_mod, kernel, iterations=1)
        image_ref_mod = cv2.erode(image_ref_mod, kernel, iterations=1)
        
        thresh = 90
        thr,img_mod = cv2.threshold(img_mod,  thresh ,255, cv2.THRESH_BINARY)
        thr2,image_ref_mod = cv2.threshold(image_ref_mod,thresh,255, cv2.THRESH_BINARY)
    
        mask = cv2.bitwise_and(image_ref_mod, img_mod)        
        diff_img = cv2.bitwise_and(diff_img_raw, cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR))

    

    '''
    cv2.namedWindow("TEST1")
    cv2.imshow("TEST1", img_mod)
    cv2.waitKey(1)
    cv2.namedWindow("TEST2")
    cv2.imshow("TEST2", image_ref_mod)
    cv2.waitKey(1)
    cv2.namedWindow("TEST3")
    cv2.imshow("TEST3", mask)
    cv2.waitKey(1)
    cv2.namedWindow("TEST4")
    cv2.imshow("TEST4", diff_img)
    cv2.waitKey(1)
    '''

     
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
    
    ref_img_name = rospy.get_param('~ref_texture_img_name', 'ref.png')

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
            pub.publish(bridge.cv2_to_imgmsg(diff_img, "bgr8")) #bgr8 #mono8
        except cv_bridge.CvBridgeError as e:
            print(e)
        r.sleep()
