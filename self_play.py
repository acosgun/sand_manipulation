#!/usr/bin/env python

import rospy
import time
import numpy as np
import math

from sandman.msg import SandActions
from std_msgs.msg import Int32MultiArray

def crop_push_points(points, crop_x_min, crop_x_max, crop_y_min, crop_y_max):
    xs = np.clip([points[0], points[2]], crop_x_min, crop_x_max)
    ys = np.clip([points[1], points[3]], crop_y_min, crop_y_max)
    points_out = [xs[0], ys[0], xs[1], ys[1]]
    return points_out

def contour_callback(msg):    
    

    # Sample a push
    X = list(msg.data)
    middle_ind = len(X)/2
    X1 = X[0:middle_ind] #Current
    X2 = X[middle_ind:] #Goal

    x1s = []
    y1s = []
    x2s = []
    y2s = []
    for i in xrange(0, middle_ind, 2):
        x1 = X1[i]
        y1 = X1[i+1]
        x2 = X2[i]
        y2 = X2[i+1]
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

            
    from random import randint

    start_pt_offset_x_min = 10
    start_pt_offset_x_max = 100

    start_pt_x = x1s[randint(0, len(x1s)/2)] + randint(start_pt_offset_x_min, start_pt_offset_x_max)
    start_pt_y = randint(min(y1s), max(y1s))
    
    s = np.random.normal(, sigma, 1000)
    
    
    

    
    # Publish the Action
    msg_actions = SandActions()
    msg_actions.header.stamp = rospy.Time.now()
    msg_actions.contour = msg.data
    msg_actions.polyreg_push.start.x = int(start_x)
    msg_actions.polyreg_push.start.y = int(start_y)
    msg_actions.polyreg_push.end.x = int(end_x)
    msg_actions.polyreg_push.end.y = int(end_y)
        
    # Wait for the message to Go Thru
    time.sleep(0.5)
    
    # Publish Command Generator Msg
    str_msg = String()
    str_msg.data = "b"
    pub.publish(str_msg)

if __name__ == '__main__':
    rospy.init_node('self_play', anonymous=True)

    pub = rospy.Publisher('/commands', String, queue_size=10)
    pub_actions = rospy.Publisher('sand_actions', SandActions, queue_size=2)
    rospy.Subscriber('contours', Int32MultiArray, contour_callback, queue_size=1)
    rospy.spin()
