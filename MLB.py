

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.optimizers import SGD
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

# divide the set into 2 sets: training set and validation set. The size of validation set is 20% of the whole data.
training_sets, test_sets, training_labels, test_labels = train_test_split(images, labels, test_size = 0.2, random_state = 30)
training_sets = training_sets/255.0
test_sets =  test_sets/255.0

classifier = Sequential()
classifier.add(Flatten())
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

classifier.fit(training_sets, training_labels, epochs = 25)

test_loss, test_accuracy = classifier.evaluate(test_sets, test_labels)
classifier.save('./classifier2.h5')
print('loss:', test_loss)
print('accuracy', test_accuracy)
