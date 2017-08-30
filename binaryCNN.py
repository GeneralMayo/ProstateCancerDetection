from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import regularizers

from scipy import ndimage
from PIL import Image
import numpy as np

import math
from keras import backend as K
from keras.utils import plot_model

def filter_image(image_tensor):
    #image_tensor[0] = detect_rotation(image_tensor[0])
    return image_tensor

batch_size = 30
minW = 160
minH = 160
all_image_types = ['ADC0','BVAL0','t2_tse_sag0','t2_tse_tra0']

image_type = all_image_types[2]

model = Sequential()
model.add(Conv2D(32, (7, 7), input_shape=(minH, minW, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#model.add(Conv2D(32, (3, 3))), 
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
		#featurewise_center=True,
		#featurewise_std_normalization=True,
        #preprocessing_function=filter_image,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        rotation_range=90,
        data_format="channels_last",
        rescale=1./255)

test_datagen = ImageDataGenerator(
	#featurewise_center=True,
	#featurewise_std_normalization=True,
    #preprocessing_function=filter_image,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    rotation_range=90,
	data_format="channels_last",
	rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'BinaryClassificationData/Train/'+image_type,
        color_mode="grayscale",
        target_size=(minH, minW), 
        batch_size=batch_size,
        class_mode='binary')  


validation_generator = test_datagen.flow_from_directory(
        'BinaryClassificationData/Test/'+image_type,
        color_mode="grayscale",
        target_size=(minH, minW),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator,
        callbacks=[ModelCheckpoint('Binary_'+image_type+'.h5', monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir='./logs', write_graph=True, write_images=False, histogram_freq=10)],
            #EarlyStopping(monitor='val_loss', patience=4)],
        steps_per_epoch=1500//batch_size,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=300//batch_size)


#model.save('Problem1_CNN.h5')
