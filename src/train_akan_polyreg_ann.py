import numpy as np
import csv
from sklearn import datasets, linear_model

def pre_processing(X, y):
    for i, val in enumerate(X):
        X_current = X[i]
        y_current = y[i]

        # Reference is the countour centroid
        x_ref = 0
        y_ref = 0
        cnt_x_ref = 0
        cnt_y_ref = 0

        
        for j in range(0,len(X[i]/2)):
            if j % 2 == 0: #index is even. X value.
                x_ref += X[i][j]
                cnt_x_ref += 1
            else:
                y_ref += X[i][j]
                cnt_y_ref += 1
        x_ref = x_ref / cnt_x_ref
        y_ref = y_ref / cnt_y_ref
        

        # Reference is the first tool point
        #x_ref = y[i][0]
        #y_ref = y[i][1]

        # Reference is the contour first point
        #x_ref = X[i][0]
        #y_ref = X[i][1]

        # No preprocessing
        #x_ref = 0
        #y_ref = 0

        X_new = []
        for j in range(0,len(X[i])):
            if j % 2 == 0: #index is even. X value.
                X_new.append(X[i][j] - x_ref)
            else:
                X_new.append(X[i][j] - y_ref)
        X[i] = X_new

        y_new = []
        for j in range(0,len(y[i])):
            if j % 2 == 0: #index is even. X value.
                y_new.append(y[i][j] - x_ref)
            else:
                y_new.append(y[i][j] - y_ref)
        y[i] = y_new

        if not (y_current == y_new).all():
            print "AAA"
        if not (X_current == X_new).all():
            print "BBB"

    return X, y


def train_regression_and_predict(regression_order, X_train, y_train, X_test):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(PolynomialFeatures(regression_order), Ridge())
    model.fit(X_train, y_train)

    model_pkl = open("../models/regression_model.pkl", 'wb')
    import pickle
    pickle.dump(model, model_pkl)
    model_pkl.close()
    
    y_pred = model.predict(X_test)
    return y_pred

def train_ann_and_predict(X_train, y_train, X_test):
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    standard_activation = 'tanh'
    final_activation = 'linear'
    num_epochs = 10000
    batch_size = 100

    input_dim = len(X_train[0])
    output_dim = len(y_train[0])

    # Initialising the ANN
    classifier = Sequential()
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = standard_activation, input_dim = input_dim))    
    #classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = final_activation))
    #classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = final_activation))
    #classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = final_activation))
    classifier.add(Dense(units = output_dim, kernel_initializer = 'uniform', activation = final_activation))
    classifier.summary()

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Keras Callbacks
    model_filepath = "../models/ann_model.hdf5"
    early_stopper_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, verbose=1, mode='auto')
    model_saver_callback = keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath= model_filepath, verbose=0, save_best_only=True) #save best model

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size = batch_size, epochs = num_epochs, shuffle=True, callbacks = [early_stopper_callback, model_saver_callback])

    # Predict!
    classifier.load_weights(model_filepath)
    y_pred = classifier.predict_on_batch(X_test)
    return y_pred


dataU_file = open('../data/dataU.dat', 'rt')
dataY_file = open('../data/dataY.dat', 'rt')

dataU_reader = csv.reader(dataU_file)
dataY_reader = csv.reader(dataY_file)

dataU = []
dataY = []

for row in dataU_reader:
    row = map(float, row)
    dataU.append(row)

for row in dataY_reader:
    row = map(float, row)
    dataY.append(row)

X_old = np.asarray(dataU)
y_old = np.asarray(dataY)

# New Data
X_new = np.loadtxt('../data/U_push_contours.txt')
X_new = X_new.astype(float)
y_new = np.loadtxt('../data/Y_push.txt')
y_new = y_new.astype(float)

use_new = True
use_old = False

if use_old == True and use_new == False:
    # Use only old data
    X = X_old
    y = y_old
elif use_old == True and use_new == True:
    # Use both new and old data
    X = np.concatenate((X_old,X_new), axis=0)
    y = np.concatenate((y_old,y_new), axis=0)
elif use_old == False and use_new == True:
    # Use only new data
    X = X_new
    y = y_new
else:
    print "No dataset selected. Exiting."
    exit(1)

#[X, y] = pre_processing(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save Scale
from sklearn.externals import joblib

train_polyreg = True

if train_polyreg == True:    
    # Train Regression 
    regression_order = 1
    y_pred = train_regression_and_predict(regression_order, X_train, y_train, X_test)
    joblib.dump(sc, "../models/regression_scale.pkl")
else:    
    # Train ANN
    y_pred = train_ann_and_predict(X_train, y_train, X_test)
    joblib.dump(sc, "../models/ann_scale.pkl")

# TEST
from utils import print_errors
print_errors(y_test, y_pred, "TEST Errors:")
print "Trained on " + str(len(X_train)) + " samples"
print "Tested on " + str(len(y_test)) + " samples"
