from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from numpy.core.defchararray import array
import tensorflow as tf
import numpy as np
import random
import decimal
import math
import os


BATCH_SIZE = 256
EPOCHS = 300

def devision_data(size):
    xdata = []
    ydata = []
    ydatalist = []
    for i in range(size):
        i1, i2 = float(decimal.Decimal(random.randrange(100, 2000))/100), float(decimal.Decimal(random.randrange(100, 2000))/100)
        y = i1 / i2 / 100
        xdata.append([i1, i2])
        ydata.append([y])
        ydatalist.append(y)

    return np.array(xdata), np.array(ydata), ydatalist

X_train, Y_train, Ytrain_list = devision_data(32000)
X_test, Y_test, Ytest_list = devision_data(12000)

def custom_activation(x):
    # smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
    # return tf.where(smallerEqualZero, K.sigmoid(x), K.sigmoid(x))
    return K.sigmoid(x)


model = Sequential([
    Dense(30, activation=custom_activation),
    Dropout(0.002),
    Dense(1, activation=custom_activation),
])

model.compile(optimizer='nadam',
              loss=tf.keras.losses.MeanAbsoluteError())

hist = model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS)

model.evaluate(X_test, Y_test)

# layer0 = [np.array([[0.000001, 0.0], [0.0, 0.000001]], dtype=np.float32), np.array([-0.000001, -0.000001], dtype=np.float32)]
# layer2 = [np.array([[33.3333], [-33.3333]], dtype=np.float32), np.array([-3.912023], dtype=np.float32)]

# model.layers[0].set_weights(layer0)
# model.layers[1].set_weights(layer2)

print('ZeroMAE*5', np.mean(Y_test))

MeanMAE5Counter = 0
for i in Ytest_list:
    MeanMAE5Counter += abs(float(i) - sum(Ytest_list)/len(Ytest_list))

print('MeanMAE*5:', MeanMAE5Counter/len(Ytest_list))