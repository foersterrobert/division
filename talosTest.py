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
        i1, i2 = float(decimal.Decimal(random.randrange(100, 2000))/100), float(decimal.Decimal(random.randrange(100, 2000))/100)
        y = i1 / i2 / 100
        xdata.append([i1, i2])
        ydata.append([y])

    return np.array(xdata), np.array(ydata)

X_train, Y_train, Ytrain_list = division_data(32000)
X_test, Y_test, Ytest_list = division_data(12000)

p = {
    'batch_size': [32, 64, 128, 256],
    'dropout': [0, 0.0002, 0.0005, 0.001],
    'epochs': [90, 270, 900, 1200],
    'optimizers': ['adam', 'nadam'],
    'losses': ['mean_absolute_error', 'mean_squared_error']
}

def division_model(x_train, y_train, params):
    def custom_activation(x):
        smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
        greaterZero = tf.greater(x, tf.constant(0.0))
        greaterFiveteen = tf.greater(x, tf.constant(15.0))
        smallerEqualFiveteen = tf.less_equal(x, tf.constant(15.0))
        return tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
                tf.where(greaterFiveteen, 1 - 1/(109.0858178 * x - 1403.359435), 
                0.03 * tf.math.log(tf.where(greaterZero, tf.where(smallerEqualFiveteen, (1000000 * x + 1), 0), 0)) + 0.5))

    model = Sequential([
        Dense(2, activation=custom_activation),
        Dropout(params['dropout']),
        Dense(1, activation=custom_activation),
    ])

    model.compile(optimizer=params['optimizers'],
                loss=params['losses'])

    out = model.fit(x_train, y_train,
            batch_size=params['batch_size'], epochs=params['epochs'])

    return out, model


division = talos.Scan(X_train, Y_train, model=division_data, params=p, experiment_name='division')

print(division)


# layer0 = [np.array([[0.000001, 0.0], [0.0, 0.000001]], dtype=np.float32), np.array([-0.000001, -0.000001], dtype=np.float32)]
# layer2 = [np.array([[33.3333], [-33.3333]], dtype=np.float32), np.array([-3.912023], dtype=np.float32)]

# model.layers[0].set_weights(layer0)
# model.layers[1].set_weights(layer2)