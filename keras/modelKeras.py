from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
from keras import Input
from keras import backend as K
from numpy.core.defchararray import array
import tensorflow as tf
import numpy as np
import random
import decimal
import math
import os

BATCH_SIZE = 256
EPOCHS = 10

def devision_data(size):
    xdata = []
    ydata = []
    ydatalist = []
    for i in range(size):
        i1, i2 = float(decimal.Decimal(random.randrange(100, 2000))/100), float(decimal.Decimal(random.randrange(100, 2000))/100)
        y = i1 / i2 / 20
        xdata.append([i1, i2])
        ydata.append([y])
        ydatalist.append(y)

    return np.array(xdata), np.array(ydata), ydatalist

X_train, Y_train, Ytrain_list = devision_data(64000)
X_test, Y_test, Ytest_list = devision_data(12000)

def custom_activation(x):
    smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
    greaterZero = tf.greater(x, tf.constant(0.0))
    greaterFiveteen = tf.greater(x, tf.constant(15.0))
    smallerEqualFiveteen = tf.less_equal(x, tf.constant(15.0))
    return tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
            tf.where(greaterFiveteen, 1 - 1/(109.0858178 * x - 1403.359435), 
            0.03 * tf.math.log(tf.where(greaterZero, tf.where(smallerEqualFiveteen, (1000000 * x + 1), 0), 0)) + 0.5))

def custom_activationLast(x):
    smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
    greaterZero = tf.greater(x, tf.constant(0.0))
    greaterFiveteen = tf.greater(x, tf.constant(15.0))
    smallerEqualFiveteen = tf.less_equal(x, tf.constant(15.0))
    x = tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
        tf.where(greaterFiveteen, 1 - 1/(109.0858178 * x - 1403.359435), 
        0.03 * tf.math.log(tf.where(greaterZero, tf.where(smallerEqualFiveteen, (1000000 * x + 1), 0), 0)) + 0.5))
    return x*5

load = input('load? y/n ')
if load == 'y':
    model = keras.models.load_model('./model/KerasHard.pth')

    while True:
        inputs = input('\ninputs: ')
        try:
            arr = np.array([[int(i.strip()) for i in inputs.split(',')]])
            print(float(model.predict(arr)*20))
        except:
            exit()

model = Sequential([
    # Input(shape=(2,)),
    Dense(2),
    Activation(custom_activation),
    Dense(1),
    Activation(custom_activationLast),
])

model.compile(optimizer='nadam',
              loss=tf.keras.losses.MeanAbsoluteError())

hist = model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS)

model.evaluate(X_test, Y_test)

layer0 = [np.array([[0.000001, 0.0], [0.0, 0.000001]], dtype=np.float32), np.array([-0.000001, -0.000001], dtype=np.float32)]
layer2 = [np.array([[33.3333], [-33.3333]], dtype=np.float32), np.array([-3.912023], dtype=np.float32)]

model.layers[0].set_weights(layer0)
model.layers[2].set_weights(layer2)

print('ZeroMAE*5', np.mean(Y_test))

MeanMAE5Counter = 0
for i in Ytest_list:
    MeanMAE5Counter += abs(float(i) - sum(Ytest_list)/len(Ytest_list))

print('MeanMAE*5:', MeanMAE5Counter/len(Ytest_list))

save = input('save? y/n ')
if save == 'y':
    model_folder_path = './model'
    file_name='KerasHard.pth'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    file_name = os.path.join(model_folder_path, file_name)
    model.save(file_name)