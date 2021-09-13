#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
from keras.datasets import mnist
from keras import models 
from keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = models.Sequential()
#two layer Dertse(fully connected) Deep Learning model. 
#10 classes softmax

network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))


#compile network
network.compile(optimizer='rmsprop',
                loss ='categorical_crossentropy',
                metrics=['accuracy'])
