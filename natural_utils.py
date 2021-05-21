# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:05:55 2021

@author: PC
"""

##### natural
import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def classes_makedirs(CLASSES, path):
    for folder in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(path, folder), exist_ok=True)
        for cls in CLASSES:
            os.makedirs(os.path.join(path, folder, cls), exist_ok=True)


def train_test_split(cls, path, val_ratio=0.1, test_ratio=0.1):
    temp_list = os.listdir(os.path.join(path, str(cls)))
    np.random.seed(42)
    np.random.shuffle(temp_list)

    temp_train_list, temp_val_list, temp_test_list = \
        np.split(np.array(temp_list), [int(len(temp_list)*(1-val_ratio-test_ratio)), int(len(temp_list) * (1 - val_ratio))])

    return temp_train_list, temp_val_list, temp_test_list


def split_classes(cls, data_list, folder, path):
    for i in range(len(data_list)):
        shutil.move(os.path.join(path, cls, data_list[i]), os.path.join(path, folder, cls))
        
        
        
        
def train_generator(path, folder, TARGET_SIZE):
    datagen = ImageDataGenerator(rescale=1./255., rotation_range=45, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    
    train_datagen = datagen.flow_from_directory(os.path.join(path, folder),
                                             class_mode='categorical',
                                             #batch_size=32,
                                             target_size=TARGET_SIZE)
    
    return train_datagen



def test_generator(path, folder, TARGET_SIZE):
    datagen = ImageDataGenerator(rescale=1./255.)
    
    test_datagen = datagen.flow_from_directory(os.path.join(path, folder),
                                             class_mode='categorical',
                                             #batch_size=32,
                                             target_size=TARGET_SIZE)
    
    return test_datagen
