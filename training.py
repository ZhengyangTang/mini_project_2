#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense 
import tensorflow as tf

from sklearn.cross_validation import train_test_split


import numpy as np
import os
import matplotlib.image as mpimg

path_name = './'

images = []
labels = []
#get images, labels, and labels are the name of file path
def get_path(path_name):
    for item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, item))

        if os.path.isdir(full_path):
            get_path(full_path)
        else:
            if item.endswith('.JPG')or item.endswith('.jpg') :
                image = mpimg.imread(full_path)
                images.append(image)
                labels.append(path_name)
    return images, labels
# turn image into arrays
images,labels = get_path(path_name)
images = np.array(images)
sizes=np.size(labels)

labelswitch=np.zeros((sizes,1))
# turn label from name of file path into class id,0 for rose, 1 for sunflower
j=0
for i in labels:
    if i.endswith('rose'):
        labelswitch[j]=0
        j = j+1
    if i.endswith('sunflower'):
        labelswitch[j]=1
        j = j+1
labels = labelswitch


training_sets, test_sets, training_labels, test_labels = train_test_split(images, labels, test_size = 0.2, random_state = 30)
training_sets = training_sets/255.0
test_sets =  test_sets/255.0
#establish sequential model
classifier = Sequential() 
 
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation = 'relu')) 

classifier.add(MaxPooling2D(pool_size = (2,2))) 

classifier.add(Flatten()) 

classifier.add(Dense(units = 128, activation='relu')) 
# output layer only contains ont node, since its either 1 or 0
classifier.add(Dense(units=1, activation='sigmoid'))
#choose optimizer, loss algorithm, and metrics
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])


classifier.fit(training_sets, training_labels, epochs = 10)

test_loss, test_accuracy = classifier.evaluate(test_sets, test_labels)
classifier.save('./classifier.h5')
print('loss:', test_loss)
print('accuracy', test_accuracy)
