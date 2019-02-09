#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:30:24 2019

@author: anilosmantur
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

import keras
import keras.utils as utils
from keras import backend as K
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

def load_model(trained=True):
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    n_classes = 10
    
    weight_path = 'weights/mnist_conv.h5'
    
    model = Sequential()
    # Convolution2D(number_filters, row_size, column_size, input_shape=(number_channels, img_row, img_col))
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    
    if trained:
        model.load_weights(weight_path)
        
    return model
 

def load_mid_model(trained=True):
    
    model = load_model(trained)
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
        
    keys = list(layer_dict.keys())
    print(keys)

    layer_name = keys[1]
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

    return intermediate_layer_model


def load_image(file_name='elephant.jpg'):
    img_rows, img_cols = 28, 28
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.show()
    
    image = cv2.resize(image, (img_rows, img_cols))
    
    plt.imshow(image)
    plt.show()
    
    x = np.transpose(image, (2, 0, 1))
    x = x.reshape(3, img_rows, img_cols, 1)
    return x

def predict_plot(model, x):
    in_out = model.predict(x)
    in_out = (in_out - in_out.min())/(in_out.max() - in_out.min())
    print(in_out.shape)
    in_out = np.transpose(in_out, (-1, 1, 2, 0))
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(in_out[i])
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
    plt.show()
    
if __name__ == '__main__':
    model = load_mid_model(trained=False)
    
    x = load_image()
    
    predict_plot(model, x)