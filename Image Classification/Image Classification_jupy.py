#!/usr/bin/env python
# coding: utf-8

# In[1]:
from keras.datasets import mnist
from keras import models
from keras import layers

# In[2]:
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# In[3]:
len(test_labels)
#test_images[0][10]
test_labels

# In[4]:
network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation= 'softmax'))

# In[5]:
network.compile(optimizer= 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

# In[6]:
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255


# In[8]:
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
test_labels

# In[9]:
network.fit(train_images, train_labels, epochs = 5, batch_size=128)


# In[10]:
test_loss, test_acc = network.evaluate(test_images, test_labels)

# In[11]:
print('test_acc:', test_acc)


# In[ ]:

