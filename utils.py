import numpy as np

def print_errors(y_pred, y_test, text):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print ""
    print "---" + text + "---"
    for i in xrange(0,y_pred.shape[1]):
        error = mean_absolute_error(y_test[:,i], y_pred[:,i])
        print error
    sq_err = mean_squared_error(y_pred, y_test)
    print "mean_squared_error: " + str(sq_err)

def sort_data_from_matrix(X):
    h,w = X.shape
    zarray = np.zeros(h*w).reshape(h,w)    
    for i in range(h):
        for j in range(w//4):
            zarray[i][4*j]   = X[i][2*j]
            zarray[i][4*j+1]   = X[i][2*j+1]
        
            zarray[i][4*j+2] = X[i][2*j+20]
            zarray[i][4*j+3] = X[i][2*j+20+1]

    return zarray
