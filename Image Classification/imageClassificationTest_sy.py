#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:52:55 2019

@author: saul
"""

import numpy as np
from vgg16 import VGG16
#from resnet50 import ResNet50
from resnet50_sy import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions

model = VGG16(include_top=True, weights='imagenet')

img_path = '/home/saul/pythontraining/imageClassification/handwriting2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

#prediction
preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

model.summary()
model.layers[-1].get_config()

#%%
model = VGG16(weights='imagenet', include_top=False)

model.summary()
model.layers[-1].get_config()

img_path = '/home/saul/pythontraining/imageClassification/handwriting2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

#%%

#RUN THIS PART FOR PREDICTION
model = ResNet50(include_top=True,weights='imagenet')
model.summary()
model.layers[-1].get_config()
img_path = '/home/saul/pythontraining/imageClassification/handwriting2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#
preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
## print: [[u'n02504458', u'African_elephant']]
#
##%%
model = ResNet50(include_top=False,weights='imagenet')
model.summary()
model.layers[-1].get_config()
img_path = '/home/saul/pythontraining/imageClassification/handwriting2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', preds)
