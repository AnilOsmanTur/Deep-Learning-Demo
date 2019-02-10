#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:12:25 2019

@author: anilosmantur
"""

import matplotlib.pyplot as plt
import numpy as np

import keras
import keras.utils as utils
from keras import backend as K
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

def first_weight_linear(i):
    img_rows, img_cols = 28, 28
    input_shape = (img_rows * img_cols,)
    n_classes = 10
    weight_paths = ['weights/mnist_linear.h5' , 'weights/mnist_linear_2.h5']
    if i == 0:
        model = Sequential([
            Dense(128, input_shape=input_shape),
            Activation('relu'),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(n_classes, activation='softmax')
        ])
    elif i == 1:
        model = Sequential([
            #Dense(128, input_shape=input_shape, activation='relu', kernel_regularizer=keras.regularizers.l2(0.2)),
            #Dropout(0.5),
            Dense(n_classes, input_shape=input_shape, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.2))
        ])
    
    model.load_weights(weight_paths[i])
    
    
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    keys = list(layer_dict.keys())
#    print(keys)
    
    layer = layer_dict[keys[0]]
#    
#    print(type(tf.Session().run(layer.weights[0])))
    
    W1 = layer.weights[0]
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    np_array = W1.eval(sess)
    
    np_array = np_array.T.reshape(-1, 28,28)
    
    return np_array

def vis_weights_linear3x3(np_array, img_rows=28, img_cols=28):
    ## Visualize sample result
    radn_n = np.random.randint(np_array.shape[0] - 9)
    plt.figure(figsize=(8, 8))
    
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(np_array[i+radn_n].reshape(img_rows, img_cols), cmap='seismic')
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
    plt.show()
    
    
def first_weight_conv():
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    n_classes = 10
    
    weight_path = 'weights/mnist_conv.h5'
    
    model = Sequential()
    # Convolution2D(number_filters, row_size, column_size, input_shape=(number_channels, img_row, img_col))
    model.add(Conv2D(16, kernel_size=(5, 5), input_shape=input_shape, padding='same'))
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
    
    model.load_weights(weight_path)
    
    
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    keys = list(layer_dict.keys())
#    print(keys)
    
    layer = layer_dict[keys[0]]
#    
#    print(type(tf.Session().run(layer.weights[0])))
    
    W1 = layer.weights[0]
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    np_array = W1.eval(sess)
    

    np_array = np_array.reshape(5, 5, 16)
    np_array = np.transpose(np_array, (2, 0, 1))
    return np_array

def plot_conv_weights(np_array):
#    np_array = (np_array - np_array.min())/(np_array.max() - np_array.min())
#    print(np_array)
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(np_array[i], cmap='seismic')
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
    plt.show()

if __name__ == '__main__':
    np_array = first_weight_conv()
    plot_conv_weights(np_array)
#    print(np_array)
