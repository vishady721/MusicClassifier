import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras import regularizers


truth_instruments = pd.read_csv('synthetic_v_natural.csv')
array = truth_instruments.values
timbres = array[:,0:12]
instruments = array[:,12]
train_timbres, test_timbres, train_instruments, test_instruments = train_test_split(timbres, instruments, test_size = 0.2, random_state = 1)

model = keras.Sequential([
    keras.layers.Dense(512, activation='elu',kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



model.fit(train_timbres, train_instruments, epochs=20)
test_loss, test_acc = model.evaluate(test_timbres,  test_instruments, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
#timbre = [[54.877,50.618,50.783,-0.586,37.078,-32.015,-5.246,-11.507,-5.105,17.845,-3.211,3.464]]
#timbre = [[35.529,-105.459,-85.876,65.314,41.501,-9.47,3.359,2.399,8.557,-23.163,-53.182,0.459]]
#timbre = [[53.905, 105.091,79.428,-19.545,39.309,-23.908,-62.024,-6.453,13.02,2.627,11.34,-2.232]]
#timbre = [[52.265, 20.441, -23.245, -35.488, 19.294, -61.816, -7.179, 23.039, -7.428, -6.272, -18.108, -30.178]]
#timbre = [[52.72, 52.711, -24.711, -7.519, 36.157, -26.431, 28.515, -31.182, -39.414, 43.54, -44.11, -14.823]]
#timbre = [[37.004, -45.005, -36.332, -30.203, 34.782, -32.041, -35.527, -24.401, -6.537, -4.953, -6.409, 8.214]]
timbre = [[49.771, 60.351, 63.458, 37.446, 63.802, -31.1, 6.959, -20.946, -55.833, -9.117, -14.269, 30.081]]
timbre_array = np.array(timbre)
timbre = (np.expand_dims(timbre_array,0))
predictions_single = probability_model.predict(timbre)
print(np.argmax(predictions_single[0]))
print(predictions_single)


