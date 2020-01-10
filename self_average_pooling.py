# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:25:15 2019

@author: foryou
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, AveragePooling2D
from quantize_valid import QuantizesValid

class SelfAveragePooling:
    def __init__(self):
        self.quantizes_valid = QuantizesValid()
        
    def data(self):
        train_x = [2.0, -8.0, 1.5, -0.5, 2.0, -8.0, 1.5, -0.5, 2.0, -8.0, 1.5, -0.5, 2.0, -8.0, 1.5, -1.5]
        train_y = [2.0, -8.0, 1.5, -0.5]
        
        train_x = self.quantizes_valid.quantizes(train_x)
        train_x = np.array(train_x).reshape(1,1,4,4)
        
        train_y = self.quantizes_valid.quantizes(train_y)
        train_y = np.array(train_y).reshape(1,1,2,2)
        return train_x, train_y
        
    def model(self):
        shape = (1,4,4)
        input = Input(shape=shape)
        x = AveragePooling2D(data_format='channels_first')(input)
        model = Model(inputs=input, outputs=x)
        return model