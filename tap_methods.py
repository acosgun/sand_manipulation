#!/usr/bin/env python

import rospy
import time
import numpy as np
import sensor_msgs.msg
import cv2
import cv_bridge
import sensor_msgs.msg
from sandman.msg import SandActions
from std_msgs.msg import Int32MultiArray

regression_model = None
regression_scale = None
ann_model = None
ann_scale = None

min_rows = 0
max_rows = 480
min_cols = 268
max_cols = 448
tool_size = 30

def crop_push_points(points):
    global crop_x_min, crop_x_max, crop_y_min, crop_y_max
    xs = np.clip([points[0], points[2]], crop_x_min, crop_x_max)
    ys = np.clip([points[1], points[3]], crop_y_min, crop_y_max)
    points_out = [xs[0], ys[0], xs[1], ys[1]]
    return points_out
        
def contour_callback(msg):  
    global min_rows, max_rows, min_cols, max_cols, tool_size
  
    msg_actions = SandActions()
    msg_actions.header.stamp = rospy.Time.now()
    '''
    if use_robs_model == False and ann_model is not None and ann_scale is not None:
        X_in = np.array(msg.data)
        X_in = np.asarray(X_in)
        X_in = X_in.reshape(1, -1)
        X_transformed = ann_scale.transform(X_in)
        from keras.models import load_model
        y_pred = ann_model.predict_on_batch(X_transformed)
        y_pred = y_pred.flatten()
        ann_points_cropped = crop_push_points(y_pred)
        msg_actions.ann_push.start.x = int(ann_points_cropped[0])
        msg_actions.ann_push.start.y = int(ann_points_cropped[1])
        msg_actions.ann_push.end.x = int(ann_points_cropped[2])
        msg_actions.ann_push.end.y = int(ann_points_cropped[3])
        
    #Rob's model
    if use_robs_model == True and ann_model is not None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.autograd import Variable
        x = Variable(torch.from_numpy(np.array(msg.data)).float())
        y = ann_model(x)
        y_pred = y.data.numpy()        
        ann_points_cropped = crop_push_points(y_pred)
        msg_actions.ann_push.start.x = int(ann_points_cropped[0])
        msg_actions.ann_push.start.y = int(ann_points_cropped[1])
        msg_actions.ann_push.end.x = int(ann_points_cropped[2])
        msg_actions.ann_push.end.y = int(ann_points_cropped[3])
        
    #Poly Regression
    if regression_model is not None and regression_scale is not None:
        X_in = np.array(msg.data)
        X_in = np.asarray(X_in)
        X_in = X_in.reshape(1, -1)
        y_pred = regression_model.predict(regression_scale.transform(X_in))
        y_pred = y_pred.flatten()
        polyreg_points_cropped = crop_push_points(y_pred)
        msg_actions.polyreg_push.start.x = int(polyreg_points_cropped[0])
        msg_actions.polyreg_push.start.y = int(polyreg_points_cropped[1])
        msg_actions.polyreg_push.end.x = int(polyreg_points_cropped[2])
        msg_actions.polyreg_push.end.y = int(polyreg_points_cropped[3])
    '''
    
    #import image from msg
    cvb = cv_bridge.CvBridge()
    img = cvb.imgmsg_to_cv2(msg, 'bgr8')
    
    #size of accessible robot workspace (refer to sand_texture.py)
    bw_img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
   
    # Average
    #TODO: 
    bw_height, bw_width = bw_img.shape
    avgX, avgY, intensity_sum = 0.0, 0.0, 0.0

    all_vals = []
    
    for i in range(0,bw_width):
        for j in range(0,bw_height):
            avgX = avgX+ i*bw_img[j,i]
            avgY = avgY+ j*bw_img[j,i]
            intensity_sum = intensity_sum+bw_img[j,i]
            all_vals.append(bw_img[j,i])

    avgX = int(avgX/intensity_sum)
    avgY = int(avgY/intensity_sum)
    #msg_actions.average_tap.end.x = avgX
    #msg_actions.average_tap.end.y = avgY


    # Max 
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(bw_img)
        
    # publish tap position
    msg_actions.maxdist_tap.end.x = int(maxLoc[0])
    msg_actions.maxdist_tap.end.y = int(maxLoc[1])


    # Importance Sampling
    probs = []
    my_sum = sum(all_vals)
    for i in range(0,len(all_vals)):
            prob = float(all_vals[i])/my_sum
            probs.append(prob)

    import numpy as np
    sample = np.random.choice(all_vals, 1, p=probs)

    #print all_vals    
    #print sample
    
    goal_i = 0
    goal_j = 0        
    for i in range(0,bw_width):
        for j in range(0,bw_height):
            val = bw_img[j,i]
            if abs(val-sample) < 0.01:
                goal_i = i
                goal_j = j
                
    msg_actions.average_tap.end.x = int(goal_i)
    msg_actions.average_tap.end.y = int(goal_j)
    
    #visualisation
    img = cv2.resize(img, (max_cols-min_cols,max_rows-min_rows), interpolation = cv2.INTER_NEAREST)
    height, width, _ = img.shape
    
    img = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img);
    img = cv2.cvtColor(img,  cv2.COLOR_GRAY2BGR)

    for i in range(0,width):
        for j in range(0,height):
	    cv2.rectangle(img, (tool_size*j, tool_size*i), (tool_size*j+tool_size, tool_size*i+tool_size), (255, 255, 255), 1)
    cv2.circle(img, (tool_size*maxLoc[0]+tool_size//2, tool_size*maxLoc[1] + tool_size//2), 5, (255, 0, 0), -1)
    cv2.circle(img, (tool_size*msg_actions.average_tap.end.x+tool_size//2, tool_size*msg_actions.average_tap.end.y + tool_size//2), 5, (0, 0, 255), -1)
    cv2.imshow("Actions", img)
    cv2.waitKey(1)


    pub_actions.publish(msg_actions)
    
    
if __name__ == '__main__':
    rospy.init_node('tap_methods', anonymous=True)
    #TODO change to models for tapping!!!!!!!!!!
    ann_model_filename = rospy.get_param('~ann_model_filename', './models/ann_model.hdf5')
    ann_scale_filename = rospy.get_param('~ann_scale_filename', './models/ann_scale.pkl')
    regression_model_filename = rospy.get_param('~regression_model_filename', './models/regression_model.pkl')
    regression_scale_filename = rospy.get_param('~regression_scale_filename', './models/regression_scale.pkl')
    
    crop_x_min = rospy.get_param('~crop_x_min', 0)
    crop_x_max = rospy.get_param('~crop_x_max', 640)
    crop_y_min = rospy.get_param('~crop_y_min', 0)
    crop_y_max = rospy.get_param('~crop_y_max', 480)

    use_robs_model = rospy.get_param('~use_robs_model', True)
    
    #Load ANN
    try:
        if use_robs_model:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.autograd import Variable
            ann_model = torch.load(ann_model_filename)
            print "Using Rob's ANN"
        else:
            ann_model_filename_akan = rospy.get_param('~ann_model_filename_akan', './models/ann_model.hdf5')
            #Load Akan's feature scaler
	    from sklearn.externals import joblib
	    ann_scale = joblib.load(ann_scale_filename)            

            #Load Akan's model
            from keras.models import load_model
            ann_model = load_model(ann_model_filename_akan)
            ann_model._make_predict_function()
            print "Using Akan's ANN"
    except:
        print "ANN model can't be loaded"

    #Load Polyreg
    try:
        # Load feature scaler
	from sklearn.externals import joblib
	regression_scale = joblib.load(regression_scale_filename)
        
        # Load model
        import pickle
	regression_model_pkl = open(regression_model_filename, 'rb')
	regression_model = pickle.load(regression_model_pkl)
        
    except:
        print "Regression model can't be loaded"
        
    pub_actions = rospy.Publisher('sand_actions_tap', SandActions, queue_size=2)
    rospy.Subscriber('texture', sensor_msgs.msg.Image, contour_callback, queue_size=1)
    rospy.spin()

