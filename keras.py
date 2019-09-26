import numpy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
import tensorflow as tf
import os
import cv2
from tensorflow.python.keras.backend import set_session

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt



#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
config =  tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )

#sess = tf.Session(config=config)
sess = tf.compat.v1.Session(config=config)
set_session(sess)



# -------- Data Augmentation --------

S=32 #64

#From Keras Documentation

trainDatagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

testDatagen = ImageDataGenerator(rescale=1./255)

trainDataset = trainDatagen.flow_from_directory(
        '/Users/tiagofonseca/Desktop/programming/champalimaudFoundation/FlyDetection/training_data/video21/training',
        target_size=(S, S),
        batch_size=32,
        class_mode='binary')

testDataset = testDatagen.flow_from_directory(
        '/Users/tiagofonseca/Desktop/programming/champalimaudFoundation/FlyDetection/training_data/video21/testing',
        target_size=(S, S),
        batch_size=32,
        class_mode='binary')


# -------- Building Convolution Neural Network -----------

#Initializing CNN
classifier = Sequential()
#Adding 1st Convolution Layer
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(S,S,3), activation='relu', padding='same'))

#Adding 1st MaxPooling Layer to reduce the size of feature map
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2) ))

#Adding 1st BatchNormalization Layer for higher Learning Rate
classifier.add(BatchNormalization())

#Adding 1st Dropout Layer to eliminate overfitting
#classifier.add(Dropout(0.2))
#Adding 2nd Convolution Layer
classifier.add(Convolution2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))

#Adding 2nd MaxPooling Layer to reduce the size of feature map
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#Adding 2nd BatchNormalization Layer for higher Learning Rate
classifier.add(BatchNormalization())

#Adding 2nd Dropout Layer to eliminate overfitting
classifier.add(Dropout(0.2))

"""
#Adding 3rd Convolution Layer
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'))

#Adding 3rd MaxPooling Layer to reduce the size of feature map
classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#Adding 3rd BatchNormalization Layer for higher Learning Rate
classifier.add(BatchNormalization())

#Adding 3rd Dropout Layer to eliminate overfitting
#classifier.add(Dropout(0.2))
"""

#Adding Flatten Layer to convert 2D matrix into an array
classifier.add(Flatten())
#Adding Fully connected layer
classifier.add(Dense(units=32,activation='relu'))

#Adding Output Layer
classifier.add(Dense(units=1,activation='sigmoid'))

print(classifier.summary())

# ----------- Compiling and Fitting the CNN to our Dataset ---------

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the CNN to images
history = classifier.fit_generator(trainDataset,
                         steps_per_epoch=700, # change to --> 8005
                         epochs=1, # change to 10
                         validation_data=testDataset,
                         validation_steps=200, # change to --> 2000
                         verbose = 1)


# ------ Visualising Accuracy and Loss w.r.t. the Epochs -------


plt.plot(history.history['acc'],'green',label='Accuracy')
plt.plot(history.history['loss'],'red',label='Loss')
plt.title('Training Accuracy & Loss')
plt.xlabel('Epoch')
plt.figure()
plt.plot(history.history['val_acc'],'green',label='Accuracy')
plt.plot(history.history['val_loss'],'red',label='Loss')
plt.title('Validation Accuracy & Loss')
plt.xlabel('Epoch')
plt.figure()



# -------- Predicting Results for some Images -------


imgOP = cv2.imread("/Users/tiagofonseca/Desktop/programming/champalimaudFoundation/FlyDetection/training_data/video21/testing/2.jpg",cv2.IMREAD_COLOR) #imgOP = cv2.imread(testingOP + directory[10])
#imgOP2 = cv2.imread("/Users/tiagofonseca/Desktop/programming/champalimaudFoundation/FlyDetection/training_data/video21/testing/2.jpg",cv2.IMREAD_COLOR) #imgOP = cv2.imread(testingOP + directory[10])
#plt.imshow(imgOP)
#cv2.imshow('NOP' , imgOP)
cv2.imwrite(os.path.join("/Users/tiagofonseca/Desktop/programming/champalimaudFoundation/FlyDetection/training_data/video21/testing" , "OP.jpg"), imgOP)
#cv2.waitKey(0)


imgOP = cv2.resize(imgOP, (S,S))
imgOP = imgOP.reshape(1,S,S,3)

imgOP = imgOP / 255


#imgOP2 = cv2.resize(imgOP2, (S,S))
#imgOP2 = imgOP2.reshape(1,S,S,3)


#pred2 = classifier.predict(imgOP2)


pred = classifier.predict(imgOP)
print("Probability that it is an ovopositor = ", "%.2f" % (1-pred))

#print(" img2 ___ Probability that it is an ovopositor = ", "%.2f" % (1-pred2))


'''

testingNOP = "/Users/tiagofonseca/Desktop/programming/champalimaudFoundation/FlyDetection/training_data/video21/testing/21_nop"
directory = os.listdir(testingNOP )
print(directory[0])

imgNOP = cv2.imread(directory[0], cv2.IMREAD_GRAYSCALE)
#plt.imshow(imgNOP)
cv2.imshow('NOP' , imgNOP)
cv2.imwrite(os.path.join("/Users/tiagofonseca/Desktop/programming/champalimaudFoundation/FlyDetection/training_data/video21/testing" , "NOP.jpg"), imgNOP)
#cv2.waitKey(0)


imgNOP = cv2.resize(imgNOP, (S,S))
imgNOP = imgNOP.reshape(1,S,S,3)

pred = classifier.predict(imgNOP)
print("Probability that it  not an ovopositor = ", "%.2f" % pred)

'''
