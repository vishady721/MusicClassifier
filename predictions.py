import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spotipy
import spotipy.util as util
import sys
import csv

scope = 'user-library-read'

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()

token = util.prompt_for_user_token(username, scope)

if token:
    #FILE NAME DEPENDING ON WHETHER YOU WANT TO RUN ON INSTRUMENTS OR NATURAL/SYNTHETIC CLASSIFIER
    truth_instruments = pd.read_csv('synthetic_v_natural.csv')
    array = truth_instruments.values
    timbres = array[:,0:12]
    instruments = array[:,12]
    train_timbres, test_timbres, train_instruments, test_instruments = train_test_split(timbres, instruments, test_size = 0.2, random_state = 1)

    model = keras.Sequential([
        keras.layers.Dense(512, activation='elu',kernel_regularizer=regularizers.l2(0.001)),
        keras.layers.Dense(5)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])



    model.fit(train_timbres, train_instruments, epochs=20)
    test_loss, test_acc = model.evaluate(test_timbres,  test_instruments, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    timbre = []
    times = []
    sp = spotipy.Spotify(auth=token)
    results = sp.audio_analysis('41L3O37CECZt3N7ziG2z7l')
    for i in range(len(results['segments'])):
        timbre.append([elem for elem in results['segments'][i]['timbre']])
        if not 'start' in results['segments'][i]:
            times.append(0)
        else:
            times.append(results['segments'][i]['start'])
    timbre_array = np.array(timbre)
    timbre = (np.expand_dims(timbre_array, 0))
    predictions = probability_model.predict(timbre)
    print(predictions)
    list_pred = []
    for elem in predictions[0]:
        list_pred.append(np.argmax(elem))

    #UNCOMMENT COLORS IF RUNNING ON INSTRUMENTS
    #colors = ['#17405a' if x == 2 else '#565289' if x == 1 else '#ae5884' if x == 4 else '#ed6e6a' if x == 3 else '#f2a93c' for x in list_pred]
    #UNCOMMENT COLORS IF RUNNING ON SYNTHETIC/NATURAL
    #colors = ['#fc766a' if x == 0 else '#5b84b1' for x in list_pred]
    
    vals = [1]*len(list_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(times, vals, 0.6, color = colors)
    
    piano_patch = mpatches.Patch(color='#17405a', label='piano')
    violin_patch = mpatches.Patch(color='#565289', label='violin')
    voice_patch = mpatches.Patch(color='#ae5884', label='voice')
    percussion_patch = mpatches.Patch(color='#ed6e6a', label='percussion')
    guitar_patch = mpatches.Patch(color='#f2a93c', label='guitar')
    
    natural_patch = mpatches.Patch(color='#fc766a', label='natural')
    synthetic_patch = mpatches.Patch(color='#5b84b1', label='synthetic')
    
    #UNCOMMENT IF RUNNING ON SYNTHETIC/NATURAL
    #plt.legend(handles = [natural_patch, synthetic_patch])
    #UNCOMMENT IF RUNNING ON INSTRUMENTS
    #plt.legend(handles = [piano_patch, violin_patch, voice_patch, percussion_patch, guitar_patch])
    
    #SAVE FIGURE TO WORKING DIRECTORY
    plt.savefig('sw_legend.png', transparent=True)
else:
    print("Can't get token for", username)

