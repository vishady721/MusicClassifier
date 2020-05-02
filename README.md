# MusicClassifier

##groundstate.csv
Contains the dataset with timbre vectors collected from the Spotify API with 0 representing guitar, 1 violin, 2 piano, 3 percussion (drums), and 4 voice.

##synthetic_v_natural.csv
Contains the dataset with timbre vectors collected from the Spotify API with 0 representing natural sound and 1 representing synthetic sound.

##groundstate.py
Adds timbre vectors to each of the ground state csv files by taking in a song ID from Spotify.

##syntheticclassifier.py and classifier.py
Uses a Neural Network that takes a timbre vector and predicts which instrument is playing and whether it is natural or has been synthesized. 

