import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
import math


def mean_pred(y_true, y_pred): #Loss function for problem error evaluation on TensorFlow
    RMSE = tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred))
    return RMSE

def num_var(X_train):
    shape = np.shape(X_train) #Shape of train array (number of events, no of variables)
    num_var = shape[1] #Show many variables on training data
    
    return num_var

def train_nn(X_train, y_train, X_test, y_test, num_var):

    # Neural Network Structure
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_var, activation='linear'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.7, seed=36),
        tf.keras.layers.Dense(20, activation='selu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # Neural Network Parameters
    lr = 0.002
    optimizer = tf.keras.optimizers.Nadam(lr=lr)
    model.compile(
        optimizer=optimizer,
        loss=mean_pred)

    # Neural Network Fitting and Validation data
    model.fit(X_train, y_train, epochs=200, steps_per_epoch=(int(len(X_train))),
            validation_data=(X_test, y_test), validation_steps=(int(len(X_test))))

    test_ans = model.predict(X_test) #predict the test data and create an array of results

    tensor_RMSE = math.sqrt(mean_squared_error(y_test, test_ans)) #get RMSE from real x predicted
    print("RMSE: " + str(tensor_RMSE))

    return model, test_ans