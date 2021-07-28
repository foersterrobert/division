from keras.backend import dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from numpy.core.defchararray import array
import tensorflow as tf
import numpy as np
import random
import decimal
import talos

def division_data(size):
    xdata = []
    ydata = []
    for i in range(size):
        i1, i2 = float(decimal.Decimal(random.randrange(100, 1000))/100), float(decimal.Decimal(random.randrange(100, 1000))/100)
        y = i1 / i2 / 100
        xdata.append([i1, i2])
        ydata.append([y])

    return np.array(xdata), np.array(ydata)

X_train, Y_train = division_data(12000)
X_test, Y_test = division_data(6000)

p = {
    'batch_size': [1, 4, 16, 32, 64],
    'dropout': [0, 0.0001, 0.0002, 0.0005],
    'epochs': [90, 240],
    'optimizers': ['rmsprop', 'ftrl', 'nadam', 'adam', 'sgd', 'adadelta', 'adamax'],
}

def division_model(x_train, y_train, x_val, y_val, params):
    def custom_activation(x):
        smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
        greaterZero = tf.greater(x, tf.constant(0.0))
        greaterFiveteen = tf.greater(x, tf.constant(15.0))
        smallerEqualFiveteen = tf.less_equal(x, tf.constant(15.0))
        x = tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
                tf.where(greaterFiveteen, 1 - 1/(109.0858178 * x - 1403.359435), 
                0.03 * tf.math.log(tf.where(greaterZero, tf.where(smallerEqualFiveteen, (1000000 * x + 1), 0), 0)) + 0.5))
        
        return x

    model = Sequential([
        Dense(2, activation=custom_activation),
        Dropout(params['dropout']),
        Dense(1, activation=custom_activation),
    ])

    model.compile(optimizer=params['optimizers'],
                loss='mean_absolute_error')

    out = model.fit(x_train, y_train,
            batch_size=params['batch_size'], epochs=params['epochs'])

    return out, model


talos.Scan(X_train, Y_train, model=division_model, params=p, experiment_name='division')