from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import decimal
import os

BATCH_SIZE = 4
EPOCHS = 90

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

X_train, Y_train, Ytrain_list = devision_data(24000)
X_test, Y_test, Ytest_list = devision_data(6000)

@tf.custom_gradient
def custom_activation(x):
    smallerEqualZero = tf.less_equal(x, tf.constant(0.0))
    greaterZero = tf.greater(x, tf.constant(0.0))
    greaterFiveteen = tf.greater(x, tf.constant(15.0))
    smallerEqualFiveteen = tf.less_equal(x, tf.constant(15.0))
    result = tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
            tf.where(greaterFiveteen, 1 - 1/(109.0858178 * x - 1403.359435), 
            0.03 * tf.math.log(tf.where(greaterZero, tf.where(smallerEqualFiveteen, (1000000 * x + 1), 0), 0)) + 0.5))

    def grad(dy):
        return dy * tf.where(smallerEqualZero, 1.359140915 * tf.math.exp(tf.where(smallerEqualZero, (x-1), 0)), 
            tf.where(greaterFiveteen, 501379254/(4596191*(501379254 * x / 4596191 - 280671887 / 200000)**2), 
            30000/(1000000*x+1)))

    return result, grad

load = input('load? h/y ')
if load == 'h':
    model = keras.models.load_model('./model/KerasHard.pth')

    while True:
        inputs = input('\ninputs: ')
        if inputs == 'w':
            for layer in model.layers:
                print(layer.get_weights())
        
        else:
            try:
                arr = np.array([[int(i.strip()) for i in inputs.split(',')]])
                print(float(model.predict(arr)*100))
            except:
                exit()

elif load == 'y':
    model = keras.models.load_model('./model/Keras.pth')

    while True:
        inputs = input('\ninputs: ')
        if inputs == 'w':
            for layer in model.layers:
                print(layer.get_weights())
        
        else:
            try:
                arr = np.array([[int(i.strip()) for i in inputs.split(',')]])
                print(float(model.predict(arr)*100))
            except:
                exit()

model = Sequential([
    Dense(2, activation=custom_activation),
    Dense(1, activation=custom_activation),
])

model.compile(optimizer='nadam',
              loss='mean_absolute_error')


hist = model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS)

# layer0 = [np.array([[0.000001, 0.0], [0.0, 0.000001]], dtype=np.float32), np.array([-0.000001, -0.000001], dtype=np.float32)]
# layer1 = [np.array([[33.3333], [-33.3333]], dtype=np.float32), np.array([-3.912023], dtype=np.float32)]

# model.layers[0].set_weights(layer0)
# model.layers[1].set_weights(layer2)

model.evaluate(X_test, Y_test)

print('ZeroMAE', np.mean(Y_test))

MeanMAE5Counter = 0
for i in Ytest_list:
    MeanMAE5Counter += abs(float(i) - sum(Ytest_list)/len(Ytest_list))

print('MeanMAE:', MeanMAE5Counter/len(Ytest_list))

save = input('save? y ')
if save == 'y':
    model_folder_path = './model'
    file_name='Keras.pth'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    file_name = os.path.join(model_folder_path, file_name)
    model.save(file_name)