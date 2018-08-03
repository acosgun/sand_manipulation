#!/usr/bin/env python

import rospy
import time
import numpy as np
from sandman.msg import SandActions
from std_msgs.msg import Int32MultiArray

scaler = None
regression_model = None
regression_scale = None
ann_model = None
ann_model2 = None
ann_scale = None
ann_new_scale = None

enable_interpolation = True
enable_feat_scaling = False

def calc_contour_error(X1,X2, middle_ind):
    err = 0
    cnt = 0
    for i in xrange(0, middle_ind, 2):
        x1 = X1[i]
        y1 = X1[i+1]
        x2 = X2[i]
        y2 = X2[i+1]

        a = np.array((x1,y1))
        b = np.array((x2,y2))
        dist = np.linalg.norm(a-b)
        err = err + dist
        cnt = cnt + 1

    avg_err = err/cnt
    return avg_err
        
def interpolate_contours(X_in):
    desired_err = 40.0    
    X = list(X_in)
    middle_ind = len(X)/2
    X1 = X[0:middle_ind]
    X2 = X[middle_ind:]
    error = calc_contour_error(X1,X2,middle_ind)
    if error < desired_err:
        return X_in
    coeff = 1/(error/desired_err)
    #print "err: " + str(error) + "coeff: " + str(coeff)
    new_X2 = []

    for i in xrange(0, middle_ind, 2):
        x1 = X1[i]
        y1 = X1[i+1]
        x2 = X2[i]
        y2 = X2[i+1]

        new_x2 = int(x2*coeff + x1*(1-coeff))
        new_y2 = int(y2*coeff + y1*(1-coeff))
        '''
        print "Coeff: " + str(coeff)
        print "x1: " + str(x1)
        print "y1: " + str(y1)
        print "x2: " + str(x2)
        print "y2: " + str(y2)
        print "new_x2: " + str(new_x2)
        print "new_y2: " + str(new_y2)
        '''
        new_X2.append(new_x2)
        new_X2.append(new_y2)
        
    return X1+new_X2

def crop_push_points(points):
    global crop_x_min, crop_x_max, crop_y_min, crop_y_max
    xs = np.clip([points[0], points[2]], crop_x_min, crop_x_max)
    ys = np.clip([points[1], points[3]], crop_y_min, crop_y_max)
    points_out = [xs[0], ys[0], xs[1], ys[1]]
    #print("Points: ", points[0], points[2], points[1], points[3])
    #print("Cropped points: ", xs[0], ys[0], xs[1], ys[1])
    #print("Cropping: ", crop_x_min, crop_x_max, crop_y_min, crop_y_max)
    return points_out
        
def contour_callback(msg):    

    msg_actions = SandActions()
    msg_actions.header.stamp = rospy.Time.now()
    msg_actions.contour = msg.data
        
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

        #Feat Scaling
        if enable_feat_scaling:            
            X_in = np.array(msg.data)
            if enable_interpolation:
                X_in = interpolate_contours(X_in)
            X_in = np.asarray(X_in)
            X_in = X_in.reshape(1, -1)
            X_in_transformed = scaler.transform(X_in)
            
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.autograd import Variable
            from numpy import zeros, newaxis
            
            #X_ann = np.array(msg.data).reshape((1, 40))            
            X_ann = X_in_transformed[:, newaxis, :]
            x = Variable(torch.from_numpy(X_ann).float())
            y = ann_model(x)
            y_pred = y.data.numpy()  
            y_pred = y_pred.flatten()
            ann_points_cropped = crop_push_points(y_pred)
            msg_actions.ann_push.start.x = int(ann_points_cropped[0])
            msg_actions.ann_push.start.y = int(ann_points_cropped[1])
            msg_actions.ann_push.end.x = int(ann_points_cropped[2])
            msg_actions.ann_push.end.y = int(ann_points_cropped[3])
        else:            
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.autograd import Variable
            from numpy import zeros, newaxis

            X_in = np.array(msg.data)
            if enable_interpolation:
                X_in = interpolate_contours(X_in)
            X_in = np.asarray(X_in)
            X_in = X_in.reshape(1, -1)
            
            #X_ann = np.array(msg.data).reshape((1, 40))            
            X_ann = X_in[:, newaxis, :]
            x = Variable(torch.from_numpy(X_ann).float())
            y = ann_model(x)
            y_pred = y.data.numpy()  
            y_pred = y_pred.flatten()
            ann_points_cropped = crop_push_points(y_pred)
            msg_actions.ann_push.start.x = int(ann_points_cropped[0])
            msg_actions.ann_push.start.y = int(ann_points_cropped[1])
            msg_actions.ann_push.end.x = int(ann_points_cropped[2])
            msg_actions.ann_push.end.y = int(ann_points_cropped[3])

        
    #Valerio's model (Pushed as Poly Reg)
    if ann_model2 is not None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.autograd import Variable
        from numpy import zeros, newaxis

        X_in = np.array(msg.data)

	enable_valerio_sort = False
	if enable_valerio_sort:
        	from utils import sort_data_from_matrix
        	X_in = sort_data_from_matrix(X_in.reshape(40,1))

        if enable_interpolation:
            X_in = interpolate_contours(X_in)
        X_in = np.asarray(X_in)
        X_in = X_in.reshape(1, -1)

        #X_ann = np.array(msg.data).reshape((1, 40))
        X_ann = X_in[:, newaxis, :]
        x = Variable(torch.from_numpy(X_ann).float())
        y = ann_model2(x)
        y_pred = y.data.numpy()  
        y_pred = y_pred.flatten()
        ann_points_cropped = crop_push_points(y_pred)
        msg_actions.polyreg_push.start.x = int(ann_points_cropped[0])
        msg_actions.polyreg_push.start.y = int(ann_points_cropped[1])
        msg_actions.polyreg_push.end.x = int(ann_points_cropped[2])
        msg_actions.polyreg_push.end.y = int(ann_points_cropped[3])
        
        
    #Poly Regression
    if False or regression_model is not None and regression_scale is not None:
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
        
    #Average and Max Dist
    X = list(msg.data)
    if enable_interpolation:
        X = interpolate_contours(X)
    middle_ind = len(X)/2
    X1 = X[0:middle_ind]
    X2 = X[middle_ind:]

    max_dist = -1
    max_dist_ind = -1
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

           
        a = np.array((x1,y1))
        b = np.array((x2,y2))
        dist = np.linalg.norm(a-b)
        if dist > max_dist:
            max_dist = dist
            max_dist_ind = i

    if max_dist_ind > -1:
        msg_actions.maxdist_push.start.x = int(X1[max_dist_ind])
        msg_actions.maxdist_push.start.y = int(X1[max_dist_ind+1])
        msg_actions.maxdist_push.end.x = int(X2[max_dist_ind])
        msg_actions.maxdist_push.end.y = int(X2[max_dist_ind+1])

    x1_mean = np.mean(x1s)
    y1_mean = np.mean(y1s)
    x2_mean = np.mean(x2s)
    y2_mean = np.mean(y2s)

    msg_actions.average_push.start.x = x1_mean
    msg_actions.average_push.start.y = y1_mean
    msg_actions.average_push.end.x = x2_mean
    msg_actions.average_push.end.y = y2_mean

    pub_actions.publish(msg_actions)
    
    
if __name__ == '__main__':
    rospy.init_node('run_net', anonymous=True)
    ann_model_filename = rospy.get_param('~ann_model_filename', './models/ann_model.hdf5')
    ann_scale_filename = rospy.get_param('~ann_scale_filename', './models/ann_scale.pkl')
    regression_model_filename = rospy.get_param('~regression_model_filename', './models/regression_model.pkl')
    regression_scale_filename = rospy.get_param('~regression_scale_filename', './models/regression_scale.pkl')

    print regression_model_filename
    
    crop_x_min = rospy.get_param('~crop_x_min', 0)
    crop_x_max = rospy.get_param('~crop_x_max', 640)
    crop_y_min = rospy.get_param('~crop_y_min', 0)
    crop_y_max = rospy.get_param('~crop_y_max', 480)

    use_robs_model = rospy.get_param('~use_robs_model', True)

    # Load feature scaler
    if enable_feat_scaling:
        from sklearn.externals import joblib
        scaler = joblib.load("/home/acrv/andrea_sand_data/ros_ws/src/sandman/models/scale.pk")
        
    #Load ANN
    try:
        if use_robs_model:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.autograd import Variable
            ann_model = torch.load('/home/acrv/andrea_sand_data/ros_ws/src/sandman/ann_v1_weights.pt') 
            print "Using ANN"
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
    except Exception as e:
	print e

    #Load Polyreg
    try:
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.autograd import Variable
        ann_model2 = torch.load('/home/acrv/andrea_sand_data/ros_ws/src/sandman/cnn_v1_weights_OLD.pt') 
        print "Using CNN"        
        
        '''
        # Load feature scaler
	from sklearn.externals import joblib
	regression_scale = joblib.load(regression_scale_filename)
        
        # Load model
        import pickle
	regression_model_pkl = open(regression_model_filename, 'rb')
	regression_model = pickle.load(regression_model_pkl)
        '''
    except:
        print "Regression model can't be loaded"
        
    pub_actions = rospy.Publisher('sand_actions', SandActions, queue_size=2)
    rospy.Subscriber('contours', Int32MultiArray, contour_callback, queue_size=1)
    rospy.spin()

