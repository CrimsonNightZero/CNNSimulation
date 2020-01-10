# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:07:50 2019

@author: foryou
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D
from quantize_valid import QuantizesValid
import os

class SelfGlobalAveragePooling:
    def __init__(self):
        self.quantizes_valid = QuantizesValid()
        self.folder_parameter = os.path.join('Parameter', 'Globalaverage_channel8')
        
    def data(self):
        train_x = list(np.arange(4,-8,-0.0625)[0:64])
        train_x.extend(np.arange(-2,8,0.125)[0:64])
        
        train_y = list(np.arange(8,-8,-0.125)[0:4])
        train_y.extend(np.arange(-8,8,0.125)[0:4])
                
        train_x_binary = self.quantizes_valid.values_to_binary(train_x)
        file_name = 'input_image'
        self.write_output(file_name, train_x_binary)
        
        train_x = self.quantizes_valid.quantizes(train_x)
        train_x = np.array(train_x).reshape(1,4,4,8)
             
        train_y = self.quantizes_valid.quantizes(train_y)
        train_y = np.array(train_y).reshape(1,len(train_y))
        
        file_name = 'input_image.txt'
        self.write_output(file_name, train_x) 
        
        np.save(os.path.join(self.folder_parameter, 'train_x'), train_x)
            
        return train_x, train_y
        
    def valid(self):
        train_x = np.load(os.path.join(self.folder_parameter, 'train_x.npy'))
        
        average_pooling = list()
        for x in range(train_x.shape[3]):
            average_pooling.append(0.0)
        print(average_pooling)
        
        for index, x in enumerate(train_x.reshape(-1, 1)):
            index = index % train_x.shape[3]
            average_pooling[index] += x
            
        for x in range(train_x.shape[3]):
            average_pooling[x] = average_pooling[x] / (train_x.shape[1] * train_x.shape[2])
        
        data_out = ''
        for x in average_pooling:
            data_out += str(x)+ '\n'
            print('output data:', x) 
        
        file_name = 'globalaverage_valid.txt'
        self.write_output(file_name, data_out)
        
    def model(self):
        shape = (4,4,8)
        input = Input(shape=shape)
        x = GlobalAveragePooling2D()(input)
        model = Model(inputs=input, outputs=x)
        return model
    
    def output(self, model, data_out):
        file_name = 'globalaverage.txt'
        self.write_output(file_name, data_out.reshape(-1, 1))
        
        data_out_binary = self.quantizes_valid.values_to_binary(data_out.reshape(-1, 1))
        print(data_out_binary)
        
        file_name = 'globalaverage'
        self.write_output(file_name, data_out_binary)
        
        print('output data', data_out)
    
    def write_output(self, file_name, values):
        if not os.path.isdir(self.folder_parameter):
                os.makedirs(r'%s/%s' % ('Parameter', 'Globalaverage_channel8'))
                
        path_file = os.path.join(self.folder_parameter, file_name)
      
        with open (path_file, 'w') as f:
            f.writelines(str(values))