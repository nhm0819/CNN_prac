# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%

import os
import numpy as np
import shutil
from utils import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator


#%%

CLASSES = ['dogs', 'cats']
path = os.path.join(os.getcwd(), "dog_cat_test_data")
os.makedirs(path+'\\validation', exist_ok=True)

allFileNames = os.listdir(os.path.join(path+'\\train'))

train_FileNames, val_FileNames = split_dog_cat(allFileNames)

classes_makedirs(path, CLASSES)

split_classes(train_FileNames, path, 'train')
split_classes(val_FileNames, path, 'validation')

#%%

datagen = ImageDataGenerator(rescale=1./255., rotation_range=45, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_datagen = datagen.flow_from_directory(os.path.join(path, 'train'),
                                             class_mode='binary',
                                             #batch_size=32,
                                             target_size=(200,200))

val_datagen = datagen.flow_from_directory(os.path.join(path, 'validation'),
                                             class_mode='binary',
                                             #batch_size=32,
                                             target_size=(200,200))

test_datagen = datagen.flow_from_directory(os.path.join(path, 'test'),
                                             class_mode='binary',
                                             #batch_size=32,
                                             target_size=(200,200))

#%%
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


#%%
n_classes = 2
input_layer = Input(shape=(200,200,3))

x = Conv2D(64, 7, padding='same', kernel_initializer='he_normal', activation='elu')(input_layer)
x = Conv2D(64, 7, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(3)(x)

x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(3)(x)

x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(3)(x)

x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(3)(x)

x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)

x = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', activation='elu')(x)
x = BatchNormalization()(x)

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)

x = Flatten()(x)
x = Dense(128, activation=LeakyReLU(alpha=0.1))(x)
x = Dense(1)(x)
output_layer = Activation('sigmoid')(x)
model = Model(input_layer, output_layer)
opt = Adam(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#%%
import datetime

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)


# history = model.fit_generator(train_datagen, steps_per_epoch=20000/32,
#                               callbacks=[early_stopping, tensorboard_callback],
#                               epochs=50,
#                               validation_data=val_datagen)

history = model.fit(train_datagen,
                    validation_data=val_datagen,
                    callbacks=[early_stopping, tensorboard_callback],
                    batch_size=32,
                    epochs=50)

#%%
save_dir = os.path.join(os.getcwd(), 'saved_model')
model_name = 'cat_dog_model_1(2).h5'


if not os.path.join(save_dir):
    os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

# cat_dog_model_1(2) : train_accuracy:0.9621, val_accuracy:0.9518, test_accuracy:0.9349

#%%
from tensorflow.keras.models import load_model
model = load_model('cat_dog_model_1(2).h5', custom_objects={'LeakyReLU':LeakyReLU})

test_loss, test_acc = model.evaluate(test_datagen, verbose=2)

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