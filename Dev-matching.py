# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:42:49 2021

@author: PC
"""
import os
import pandas as pd
import numpy as np

classes = ['dog', 'elephant', 'giraffe','guitar','horse','house','person']
path = os.path.join(os.getcwd(), 'train')
#%%

train_data = pd.DataFrame(columns=['filename', 'label'])

for cls in classes:
    temp_data = pd.DataFrame()
    temp_list = os.listdir(os.path.join(path,str(cls)))
    temp_data['filename'] = temp_list
    temp_data['label'] = [cls] * len(temp_list)
    train_data = train_data.append(temp_data)

train_data['filename'] = train_data['label'] +'\\'+ train_data['filename']

X = train_data
y = train_data['label']
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
TARGET_SIZE = (227,227)
BATCH_SIZE = 32

train_generator = ImageDataGenerator(rescale=1./255.,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   horizontal_flip=True
                                   )

test_generator = ImageDataGenerator(rescale=1./255.)

#%%
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
n_classes = len(classes)
input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)

model = DenseNet121(weights=None, input_shape=input_shape, classes=n_classes)

opt = Adam(lr=0.0005)

model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

#%%
import tensorflow as tf
import datetime
import tensorflow.keras.callbacks as callbacks

early_stopping = callbacks.EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)

#%%
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 3, shuffle = True) 
#%%
cvscores = []
fold = 1
model_name=''

for _ in range(33):
    for train_index, val_index in skf.split(X, y):
        training_data = X.iloc[train_index]
        validation_data = X.iloc[val_index]
        print ('Fold: ',fold)
        
        train_datagen = train_generator.flow_from_dataframe(training_data,
                                                            directory=path,
                                                            x_col="filename",
                                                            y_col="label",
                                                            class_mode="categorical",
                                                            target_size=TARGET_SIZE,
                                                            batch_size=BATCH_SIZE,
                                                            shuffle=True)
        
        validation_datagen = test_generator.flow_from_dataframe(dataframe=validation_data,
                                                                      directory=path,
                                                                      x_col="filename",
                                                                      y_col="label",
                                                                      class_mode = "categorical",
                                                                      target_size=TARGET_SIZE,
                                                                      batch_size=8,
                                                                      shuffle=True)
        
        if(os.path.isfile(model_name+'.h5')):
            model.load_weights(model_name)
        
        history = model.fit(train_datagen,
                            epochs=20,
                            callbacks=[early_stopping],
                            validation_data=validation_datagen)
        
        
        
        # Save each fold model
        model_name = 'model_DenseNet2_fold_'+str(fold)+'.h5'
        model.save(model_name)
        
        model.load_weights(model_name)
        # results = model.evaluate(validation_datagen)
        # results = dict(zip(model.metrics_names, results))
    	
        # cvscores.append(results['accuracy'])
        # cvscores.append(results['loss'])
    	
        tf.keras.backend.clear_session()
        
        # evaluate the model
        scores = model.evaluate(validation_datagen, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        
        ## save the probability prediction of each fold in separate csv file
        # proba = model.predict(test_datagen, batch_size=None, steps=1)
        # labels=[np.argmax(pred) for pred in proba]
        # csv_name= 'submission_CNN_keras_aug_Fold'+str(fold)+'.csv'
        # create_submission(predictions=labels,path=csv_name)
        
        fold += 1


#%%
# model = DenseNet121(weights=None, input_shape=input_shape, classes=n_classes)
# model.load_weights('model_DenseNet_fold_22.h5')

test_generator = ImageDataGenerator(rescale=1./255.)

test_datagen = test_generator.flow_from_directory(os.path.join(os.getcwd(), 'test'),
                                                  # only read images from `test` directory
                                                  # don't generate labels
                                                  class_mode=None,
                                                  # don't shuffle
                                                  shuffle=False,
                                                  # use same size as in training
                                                  target_size=TARGET_SIZE)

preds = np.argmax(model.predict(test_datagen), axis=1)

result = pd.DataFrame(preds, columns=['answer'])
result.to_csv(os.path.join(os.getcwd(), 'result.csv'), sep=',')

#%%
plt.plot(cvscores)
plt.ylabel('Validation Accuracy')
plt.xlabel('epoch')
