# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:45:17 2021

@author: PC
"""

import os
import numpy as np
import shutil

def split_dog_cat(allFileNames, val_ratio = 0.2):
    np.random.shuffle(allFileNames)

    dog_list = []
    cat_list = []

    for i in range(len(allFileNames)):
        if(allFileNames[i].startswith('dog')):
            dog_list.append(allFileNames[i])
        elif(allFileNames[i].startswith('cat')):
            cat_list.append(allFileNames[i])

    dog_train_FileNames, dog_val_FileNames = np.split(np.array(dog_list), [int(len(dog_list) * (1 - val_ratio))])
    cat_train_FileNames, cat_val_FileNames = np.split(np.array(cat_list), [int(len(cat_list) * (1 - val_ratio))])

    train_FileNames = np.concatenate((dog_train_FileNames, cat_train_FileNames), axis=0)
    val_FileNames = np.concatenate((dog_val_FileNames, cat_val_FileNames), axis=0)
    np.random.shuffle(train_FileNames)
    np.random.shuffle(val_FileNames)

    return train_FileNames, val_FileNames


def classes_makedirs(path, CLASSES):
    for split_folder in ['train', 'validation']:
        for cls in CLASSES:
            os.makedirs(os.path.join(path, split_folder, cls), exist_ok=True)


def split_classes(FileNames, path, train_or_validation):
    for i in range(len(FileNames)):
        if(FileNames[i].startswith('cat')):
            shutil.move(os.path.join(path, 'train', FileNames[i]), os.path.join(path, str(train_or_validation), 'cats'))
        elif (FileNames[i].startswith('dog')):
            shutil.move(os.path.join(path, 'train', FileNames[i]), os.path.join(path, str(train_or_validation), 'dogs'))

