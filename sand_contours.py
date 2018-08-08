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

enable_box_chop = True

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
        #draw_push(img, sand_actions_msg.ann_push, brush_width, (12,255,255), "ANN", (50, 30)) #Yellow
        draw_push(img, sand_actions_msg.polyreg_push, brush_width, (0,255,125), "CNN", (50, 50)) #GRN
        #draw_push(img, sand_actions_msg.average_push, brush_width, (12,120,255), "Avg Contour", (50, 70))
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
    
    if box is None:
        print "BOX IS NONE!!"
    else:
        cur_thresh_img = get_thresh_img(img.copy(), box)
        ref_thresh_img = get_thresh_img(image_ref.copy(), box)

        out_box = find_feasible_contours(box, cur_thresh_img, ref_thresh_img, enable_box_chop)
        if out_box is None:
            print "Out box is None!"
            return
        box = out_box
        
        #curCont = get_contours(img.copy(), box, True)
        #refCont = get_contours(image_ref.copy(), box, False)        
        #img_contours = sample_contours(curCont, 10, True)
        #ref_contours = sample_contours(refCont, 10, True)

        curCont = get_contours(img.copy(), box, False)
        refCont = get_contours(image_ref.copy(), box, False)        
        img_contours = sample_contours(curCont, 10, True)
        ref_contours = sample_contours(refCont, 10, True)
        
        if curCont is not None:
            cv2.drawContours(this_ref, curCont, -1, (255,0,0), 0)
    
        if refCont is not None and len(refCont) > 0:
            cv2.drawContours(this_ref, refCont, -1, (0,0,255), 0)
        
        cv2.rectangle(img, (box.x, box.y), (box.bot_right_x, box.bot_right_y), (255,0,0))
        
        if img_contours is not None:
            final_curr = img_contours
            for contour in img_contours:
                #for p in contour:
                #cv2.circle(img, tuple(p[0]), 2, [255, 0, 0], thickness=2)
                for idx,p in enumerate(contour):
                    cv2.circle(img, tuple(p[0]), 2, [255, 0, 0], thickness=2)
                    cv2.putText(img, str(idx), tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], thickness=2)

        
        if ref_contours is not None:
            final_ref = ref_contours
            for contour in ref_contours:
                #for p in contour:
                for idx,p in enumerate(contour):
                    cv2.circle(img, tuple(p[0]), 2, [0, 0, 255], thickness=2)
                    cv2.putText(img, str(idx), tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], thickness=2)
                                        
        if ref_contours is None or img_contours is None:
            print("I think I am done! DO NOT PRESS a ON COMMAND GENERATOR - it wouldn't make sense!")
            
        draw_candidate_pushes(img)
            
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 1280, 0)
        cv2.imshow("Image", img)

        cv2.namedWindow("Image_Ref")
        cv2.moveWindow("Image_Ref", 600, 0)
        cv2.imshow("Image_Ref", this_ref)
        cv2.waitKey(1)

def sample_contours(contours, num, verbose):
    if contours is not None and len(contours) > 0:
        contour = contours[0]
        samples = []
        if len(contour) > num:
            length = float(len(contour))
            for i in range(num):
                sample = contour[int(math.ceil(i * length / num))]
                samples.append(sample)

            assert len(samples) == num
            samples = sorted(samples, key=lambda x: x[0][1], reverse = False)
                             
            return [samples]
        else:
            if verbose:
                print("NOT ENOUGH POINTS: less than 10 points in samples")
            return None
    else:
        if verbose:
            print("NOT ENOUGH POINTS: less than 1 point in contours")
        return None

def get_thresh_img(image, roi):
    MIN_PIXELS = 50
    lower = np.array([0, 0, 100])
    upper = np.array([255, 50, 255])

    cropped = image[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    thresh  = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    return thresh        

def find_feasible_contours(roi, cur_thresh_img, ref_thresh_img, enable_box_chop):

    if enable_box_chop is False:
        return roi
    
    MIN_PIXELS = 50
    init_desired_box_height = 100
    end_desired_box_height = 200
    step_h = 20
    num_h_samples = 20
    
    if enable_box_chop:

        for desired_box_height in xrange(init_desired_box_height,end_desired_box_height,step_h):
            if roi.h < desired_box_height:
                return roi
            
            for i in xrange(num_h_samples):            
                #sample new y_top                
                from random import randint
                y_min = roi.y
                y_max = roi.y + roi.h - desired_box_height
                new_y_top = randint(y_min, y_max)                
                
                #get chopped images
                cur_chopped_img = cur_thresh_img[new_y_top:new_y_top+desired_box_height,:]
                ref_chopped_img = ref_thresh_img[new_y_top:new_y_top+desired_box_height,:]
                
                #find contours in the chopped images
                _, cur_contours, _= cv2.findContours(cur_chopped_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                _, ref_contours, _= cv2.findContours(ref_chopped_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                if cur_contours is None or ref_contours is None:
                    continue
                else:            
                    cur_contours = filter_contours(cur_contours, MIN_PIXELS)
                    ref_contours = filter_contours(ref_contours, MIN_PIXELS)
                    if cur_contours is None or ref_contours is None:
                        continue
                    else:
                        new_roi = Box(roi.x, new_y_top, roi.w, desired_box_height)
                        
                        cur_contours = remove_border(cur_contours, new_roi, thresh=1)
                        ref_contours = remove_border(ref_contours, new_roi, thresh=1)
                        
                        if cur_contours is None or ref_contours is None:
                            continue
                        else:
                            cur_contours = sample_contours(cur_contours, 10, False)
                            ref_contours = sample_contours(ref_contours, 10, False)
                            if cur_contours is None or ref_contours is None:
                                continue
                            else:                            
                                return new_roi
    return roi

def get_contours(image, roi, draw_test):
    MIN_PIXELS = 50
    lower = np.array([0, 0, 100])
    upper = np.array([255, 50, 255])

    cropped = image[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    thresh  = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    _, contours, _= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if draw_test:
        cv2.namedWindow("TEST")
        cv2.imshow("TEST", thresh)
                
    if contours is not None and len(contours) > 0:
        contours = filter_contours(contours, MIN_PIXELS)
        if contours is not None:
            contours = remove_border(contours, roi, thresh=1)#was 5
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

    '''
    padding = 0
    box.x = box.x - padding
    box.h = box.h + 2*padding
    '''
    return box

def centroid_dist(box, largest):
    dist = np.sqrt((largest.cx-box.cx)**2 + (largest.cy-box.cy)**2)
    dist = dist/np.sqrt(largest.w**2 + largest.h**2)
    return dist


def filter_contours(contours, thresh):
    if contours is None:
        return None
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
    THRESH     = 35
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
    pub = rospy.Publisher('contours', Int32MultiArray, queue_size=1)

    
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
            final_ref = None
            final_curr = None
        r.sleep()
