#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt


# In[2]:


data_from_dir = 'C:/Users/young/Desktop/R/cats-dogs'
base_dir = base_dir='C:/Users/young/Desktop/R/kaggle_cat'

train_dir='C:/Users/young/Desktop/R/kaggle_cat/train'
val_dir='C:/Users/young/Desktop/R/kaggle_cat/val'
test_dir='C:/Users/young/Desktop/R/kaggle_cat/test'

train_cats_dir=f'C:/Users/young/Desktop/R/kaggle_cat/train/cats'
train_dogs_dir='C:/Users/young/Desktop/R/kaggle_cat/train/dogs'
val_cats_dir='C:/Users/young/Desktop/R/kaggle_cat/val/cats'
val_dogs_dir='C:/Users/young/Desktop/R/kaggle_cat/val/dogs'
test_cats_dir='C:/Users/young/Desktop/R/kaggle_cat/test/cats'
test_dogs_dir='C:/Users/young/Desktop/R/kaggle_cat/test/dogs'


# In[3]:


from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale = 1/255,
                             zoom_range= 0.2)
data_gen = ImageDataGenerator(rescale = 1/255)

train_gen = train_data_gen.flow_from_directory(train_dir,
                                        target_size=(150,150),
                                        batch_size=20,
                                        class_mode='binary')
val_gen = data_gen.flow_from_directory(val_dir,
                                        target_size=(150,150),
                                        batch_size=20,
                                        class_mode='binary')
test_gen = data_gen.flow_from_directory(test_dir,
                                        target_size=(150,150),
                                        batch_size=20,
                                        class_mode='binary')


# In[4]:


from keras import models, layers

model = models.Sequential()

model.add(layers.Conv2D(32, activation='relu', input_shape=(150,150,3), kernel_size=3))
model.add(layers.MaxPool2D(2))
model.add(layers.Conv2D(32, activation='relu', kernel_size=3))
model.add(layers.MaxPool2D(2))
model.add(layers.Conv2D(32, activation='relu',  kernel_size=3))
model.add(layers.MaxPool2D(2))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=['acc'])


# In[5]:

# if got error about pil~~, pip install image can solve that
history = model.fit_generator(train_gen,
                             steps_per_epoch=100,
                             epochs = 10,
                             validation_data= val_gen,
                             validation_steps=50)


# In[7]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# In[8]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# In[ ]:




