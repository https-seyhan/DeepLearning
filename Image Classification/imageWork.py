#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:37:33 2019

@author: saul
"""
import os

from keras.preprocessing import image

os.chdir('/home/saul/pythontraining/NLP')

img1 = image.load_img('23_July_1915.jpg', target_size=(150, 150))
print(img1)

x1 = image.img_to_array(img1)

x1 = x1.reshape((1,) + x1.shape)

print(x1)

i = 0

for batch in datagen.flow(x1, batch_size =1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i +=1
    if i % 4 == 0:
        break
plt.show()
