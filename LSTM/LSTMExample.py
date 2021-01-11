#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
#Keras and LSTM

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.utils.vis_utils import plot_model

max_features = 20000
maxlen = 80
batch_size = 32

#Prepare dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print(len(X_train))
print(X_train[1])
#print(X_train[1][188])

#LSTM
model = Sequential()
model.add(Embedding(max_features, 128, dropout= 0.2))

model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.summary())

#Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training
model.fit(X_train, y_train, batch_size=batch_size, verbose=1, nb_epoch=1, validation_data=(X_test, y_test))

# Evaluation
score = model.evaluate(X_test, y_test, batch_size = batch_size)
print("Test Results :", zip(model.metrics_names, score))
plot_model(model, to_file='LSTM.png', show_shapes=True)
