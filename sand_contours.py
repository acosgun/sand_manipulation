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

min_blue = np.array([110,0,0])
max_blue = np.array([255,255,56])

sand_actions_msg = None
final_contours = None
final_ref = None
final_curr = None
image_ref = None
image_curr = np.zeros_like(image_ref)

class Box():
    def __init__(self, x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.cx = x+(w/2)
        self.cy = y+(h/2)
        self.bot_right_x = x+w
        self.bot_right_y = y+h


def draw_push(img, push_action, brush_width, color, text, text_pos):
    pt1 = (push_action.start.x, push_action.start.y)
    pt2 = (push_action.end.x, push_action.end.y)
    if pt1[0] is not 0 or pt1[0] is not 0 or pt2[0] is not 0 or pt2[1] is not 0:
        cv2.arrowedLine(img, pt1, pt2, color, brush_width)
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, text_pos, fontface, 0.7, color, 2)

def draw_candidate_pushes(img):
    global sand_actions_msg
    brush_width = 4
    if sand_actions_msg is not None:
        draw_push(img, sand_actions_msg.ann_push, brush_width, (12,255,255), "Neural Net", (50, 30)) #Yellow
        draw_push(img, sand_actions_msg.polyreg_push, brush_width, (0,255,125), "Poly Regression", (50, 50))
        draw_push(img, sand_actions_msg.average_push, brush_width, (12,120,255), "Avg Contour", (50, 70))
        draw_push(img, sand_actions_msg.maxdist_push, brush_width, (255, 255, 0), "Max Contour", (50, 90))

def sand_actions_callback(msg):
    global sand_actions_msg
    sand_actions_msg = msg
        
def image_capture(msg):
    global image_ref, final_ref, final_curr, save_new
    
    cvb = cv_bridge.CvBridge()
    # Convert into opencv matrix
    img = cvb.imgmsg_to_cv2(msg, 'bgr8')
    im_size = img.shape

    if save_new:
        print("Storing Ref Image")
        image_ref = img
        cv2.imwrite(ref_img_name,img)
        # cv2.imshow("ref",image_ref)
        # cv2.waitKey(0)
        save_new = False
    this_ref = image_ref.copy()
    box = get_roi(img.copy())
    if box is not None:
        img_contours = sample_contours(get_contours(img.copy(), box))
        ref_contours = sample_contours(get_contours(image_ref.copy(), box))

        if ref_contours is not None:
            final_ref = ref_contours
            for contour in ref_contours:
                #for p in contour:
                for idx,p in enumerate(contour):
                    cv2.circle(this_ref, tuple(p[0]), 2, [255, 0, 0], thickness=2)
                    #cv2.putText(this_ref, str(idx), tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], thickness=2)

            cv2.rectangle(this_ref, (box.x, box.y), (box.bot_right_x, box.bot_right_y), (255,0,0))
        

        if img_contours is not None:
            final_curr = img_contours
            for contour in img_contours:
                #for p in contour:
                    #cv2.circle(img, tuple(p[0]), 2, [255, 0, 0], thickness=2)
                for idx,p in enumerate(contour):
                    cv2.circle(img, tuple(p[0]), 2, [255, 0, 0], thickness=2)
                    #cv2.putText(img, str(idx), tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], thickness=2)

            cv2.rectangle(img, (box.x, box.y), (box.bot_right_x, box.bot_right_y), (255,0,0))

        if ref_contours is None or img_contours is None:
            print("I think I am done! DO NOT PRESS a ON COMMAND GENERATOR - it wouldn't make sense!")
            
        draw_candidate_pushes(img)
            
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 1280, 0)
        cv2.imshow("Image", img)

        cv2.namedWindow("Image_Ref")
        cv2.moveWindow("Image_Ref", 660, 0)
        cv2.imshow("Image_Ref", this_ref)
        cv2.waitKey(1)

def sample_contours(contours):
    #if len(contours) > 0:
        #contour = contours[0]
        #samples = []
        #if len(contour) > 10:
            #length = float(len(contour))
            #for i in range(10):
                #sample = contour[int(math.ceil(i * length / 10))]
                #samples.append(sample)

            #assert len(samples) == 10

            #return [samples]
        #else:
            # print("NOT ENOUGH POINTS")
            #return None
    #return None
    if contours is not None and len(contours) > 0:
        contour = contours[0]
        samples = []
        if len(contour) > 10:
            length = float(len(contour))
            for i in range(10):
                sample = contour[int(math.ceil(i * length / 10))]
                samples.append(sample)

            assert len(samples) == 10

            return [samples]
        else:
            print("NOT ENOUGH POINTS: less than 10 points in samples")
            return None
    else:
        print("NOT ENOUGH POINTS: less than 1 point in contours")
        return None

def get_contours(image, roi):
    MIN_PIXELS = 50
    lower = np.array([0, 0, 100])
    upper = np.array([255, 50, 255])

    cropped = image[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    thresh  = cv2.inRange(hsv, lower, upper)

    _, contours, _= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours = filter_contours(contours, MIN_PIXELS)
        if contours is not None:
            contours = remove_border(contours, roi, thresh=5)

    return contours
   
def remove_border(contours, roi, thresh=2):
    new_contours = []
    for contour in contours:
        for p in contour:
            x = p[0][0]; y = p[0][1]
            if  x > thresh and y > thresh and x < roi.w - thresh and y < roi.h - thresh:
                new_contours.append([[x + roi.x, y + roi.y]])
    new_contours = np.asarray(new_contours)
    return [new_contours]


def get_box(contour):
    box = Box(*cv2.boundingRect(contour))
    return box

def centroid_dist(box, largest):
    dist = np.sqrt((largest.cx-box.cx)**2 + (largest.cy-box.cy)**2)
    dist = dist/np.sqrt(largest.w**2 + largest.h**2)
    return dist


def filter_contours(contours, thresh):
    # Sort contours and filter
    largestContours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largestContours = [contour for contour in largestContours if cv2.contourArea(contour) > thresh]
    # return largestContours
    # return random.sample(largestContours, 1)
    if len(largestContours) > 0:
        return [largestContours[0]]
    return None


def get_roi(img):
    # Binary threshold value
    THRESH     = 50
    MIN_PIXELS = 200 #500 # Min size for cont
    MAX_DIST   = 3

    diff          = cv2.subtract(img, image_ref)
    diff          = cv2.cvtColor(diff,  cv2.COLOR_BGR2GRAY)
    _, bin_diff   = cv2.threshold(diff, THRESH, 255,  cv2.THRESH_BINARY)

    _, contours, _= cv2.findContours(bin_diff.copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        largestContours = filter_contours(contours, MIN_PIXELS)
        #if len(largestContours) > 0:
        if largestContours != None:
            largestContour = largestContours[0]

            largest_box = get_box(largestContour)

            for contour in largestContours[1:]:
                box = get_box(contour)
                if centroid_dist(box, largest_box) < MAX_DIST:
                    if (largest_box.x >  box.x):
                        largest_box.x = box.x
                    if (largest_box.y >  box.y):
                        largest_box.y = box.y

                    if (largest_box.bot_right_x <  box.bot_right_x):
                        largest_box.bot_right_x =  box.bot_right_x

                    if (largest_box.bot_right_y <  box.bot_right_y):
                        largest_box.bot_right_y =  box.bot_right_y

                    largest_box.w = largest_box.bot_right_x - largest_box.x
                    largest_box.h = largest_box.bot_right_y - largest_box.y

            cv2.rectangle(img, (largest_box.x, largest_box.y), (largest_box.bot_right_x, largest_box.bot_right_y), (255,0,0))
            cv2.drawContours(img, largestContours, -1, (0,255,0), 0)
            
            cv2.namedWindow("Contours")
            cv2.moveWindow("Contours", -50, 0)
            cv2.imshow("Contours", img)

            #cv2.imshow("Diff", bin_diff)
            #cv2.imshow("Ref", image_ref)
            cv2.waitKey(1)
            return largest_box
        else:
            print("No large contour found!")
    else:
        print("No contours found!")


if __name__ == '__main__':
    global save_new
    rospy.init_node('contour_detector',anonymous=True) # node name
    
    ref_img_name = rospy.get_param('~ref_img_name', 'ref.png')

    save_new = raw_input("Save new image? y/n: ") == 'y'

    if save_new == False:
        image_ref = cv2.imread(ref_img_name)

    image_sub = rospy.Subscriber('/camera/rgb/image_raw', sensor_msgs.msg.Image, image_capture, queue_size=1) #listen robot position
    actions_sub = rospy.Subscriber('/sand_actions', SandActions, sand_actions_callback, queue_size=1)
    pub = rospy.Publisher('contours', Int32MultiArray, queue_size=10)

    
    # Publish velocity at 100Hz.
    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        if final_ref is not None and final_curr is not None:
            final_contours = Int32MultiArray()
            data = []
            
            for p in final_curr[0]:
                data.append(p[0][0])
                data.append(p[0][1])
            for p in final_ref[0]:
                data.append(p[0][0])
                data.append(p[0][1])


            final_contours.data = data
            pub.publish(final_contours)
        r.sleep()
