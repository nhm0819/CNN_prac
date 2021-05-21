# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:04:59 2021

@author: Hongmin
"""

import os
import numpy as np
import shutil
from natural_utils import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
Data counts

- airplane : 727
- car      : 968
- cat      : 885
- dog      : 702
- flower   : 843
- fruit    : 1000
- motorbike: 788
- person   : 986

'''

CLASSES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
path = os.path.join(os.getcwd(), "natural_images")

#%% Preprocessing

classes_makedirs(CLASSES, path)

for i in range(len(CLASSES)):
    train_list, validation_list, test_list = train_test_split(CLASSES[i], path, val_ratio=0.1, test_ratio=0.1)
    split_classes(CLASSES[i], train_list, 'train', path)
    split_classes(CLASSES[i], validation_list, 'validation', path)
    split_classes(CLASSES[i], test_list, 'test', path)


for i in range(len(CLASSES)):
    os.removedirs(os.path.join(path, CLASSES[i]))

#%%

TARGET_SIZE = (128,128)

# datagen = ImageDataGenerator(rescale=1./255., rotation_range=45, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)


# train_datagen = datagen.flow_from_directory(os.path.join(path, 'train'),
#                                              class_mode='categorical',
#                                              #batch_size=32,
#                                              target_size=(128,128))

# val_datagen = datagen.flow_from_directory(os.path.join(path, 'validation'),
#                                              class_mode='categorical',
#                                              #batch_size=32,
#                                              target_size=(128,128))

# test_datagen = datagen.flow_from_directory(os.path.join(path, 'test'),
#                                              class_mode='categorical',
#                                              #batch_size=32,
#                                              target_size=(128,128))

train_datagen = train_generator(path, 'train', TARGET_SIZE)
validation_datagen = test_generator(path, 'validation', TARGET_SIZE)
test_datagen = test_generator(path, 'test', TARGET_SIZE)


#%%
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

#%%
n_classes = 8
input_layer = Input(shape=(128,128,3))

x = Conv2D(64, 7, padding='same', kernel_initializer='he_normal', activation='elu')(input_layer)
x = Conv2D(64, 7, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)

x = Dropout(0.4)(x)
x = GlobalAveragePooling2D()(x)

x = Flatten()(x)
x = Dense(128, activation='elu')(x)
x = Dense(n_classes)(x)
output_layer = Activation('softmax')(x)
model = Model(input_layer, output_layer)
opt = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#%%
import datetime
os.makedirs(os.path.join("logs", "natural"), exist_ok=True)
logdir = os.path.join("logs", "natural", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)


history = model.fit(train_datagen,
                    validation_data=validation_datagen,
                    callbacks=[early_stopping, tensorboard_callback],
                    batch_size=20,
                    epochs=20)

#%%
save_dir = os.path.join(os.getcwd(), 'saved_model')
model_name = 'natural_model_5.h5'


if not os.path.join(save_dir):
    os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, model_name)

model.save(model_path)


#%%
test_loss, test_acc = model.evaluate(test_datagen, verbose=2)

# model 1 : loss:        - accuracy: 0.94   ( train : val : test = 9 : 0 : 1)
# model 3 : loss: 0.2214 - accuracy: 0.9235 ( train : val : test = 8 : 1 : 1)
# model 4 : loss: 0.19   - accuracy: 0.9380 ( train : val : test = 8 : 1 : 1)
# model 5 : 
#%%

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

