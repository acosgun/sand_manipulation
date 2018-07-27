def print_errors(y_pred, y_test, text):
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print ""
    print "---" + text + "---"
    for i in xrange(0,y_pred.shape[1]):
        error = mean_absolute_error(y_test[:,i], y_pred[:,i])
        print error
    sq_err = mean_squared_error(y_pred, y_test)
    print "mean_squared_error: " + str(sq_err)
