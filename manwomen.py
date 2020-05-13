#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()
model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()


# In[2]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'menwomen/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'menwomen/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


# In[3]:


model.fit_generator(
    training_set, 
    steps_per_epoch=len(training_set),
    validation_data=test_set, 
    validation_steps=len(test_set), 
    epochs=25
    )

#Accuracy around 92%
# In[31]:


model.save('menwomenmodel.h5')

# load model
from keras.models import load_model
m = load_model('menwomenmodel.h5')
from keras.preprocessing import image


# In[32]:


test_image = image.load_img('vivekphotot.jpeg', target_size=(64,64))


# In[33]:


test_image = image.img_to_array(test_image)


# In[34]:


import numpy as np


# In[35]:


test_image = np.expand_dims(test_image, axis=0)


# In[36]:


result = m.predict(test_image)


# In[37]:


if result[0][0] == 1.0:
    print('women')
else:
    print('men')


# In[38]:


training_set.class_indices







