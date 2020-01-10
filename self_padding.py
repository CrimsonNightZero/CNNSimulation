# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:21:48 2019

@author: foryou
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:07:50 2019

@author: foryou
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, ZeroPadding2D
from quantize_valid import QuantizesValid
from read_data import ReadData
import os

class SelfPadding:
    def __init__(self, padding):
        self.quantizes_valid = QuantizesValid()
        self.read_data = ReadData()
        self.padding = padding
        if "back" in self.padding:
            self.stride = 2
        else:
            self.stride = 1
        self.folder_parameter = os.path.join('Parameter', 'Padding_' + self.padding)
        
    def data(self):
        train_x = list(np.arange(4,-4,-0.125)[0:64])
        train_x.extend(np.arange(-4,4,0.125)[0:64])
        
        if "valid" in self.padding :
            train_y = list(np.arange(8,-8,-0.125)[0:32])
            train_y.extend(np.arange(-8,8,0.125)[0:32])
                
        elif "same" in self.padding:
            train_y = list(np.arange(10,-10,-0.125)[0:144])
            train_y.extend(np.arange(-10,10,0.125)[0:144])
                
        elif "back" in self.padding:   
            train_y = list(np.arange(8,-8,-0.125)[0:100])
            train_y.extend(np.arange(-8,8,0.125)[0:100])
        
        train_x_binary = self.quantizes_valid.values_to_binary(train_x)
        file_name = 'input_image'
        self.write_output(file_name, train_x_binary)
        
        train_x = self.quantizes_valid.quantizes(train_x)
        train_x = np.array(train_x).reshape(1,4,4,8)
        
        train_y = self.quantizes_valid.quantizes(train_y)
        if "valid" in self.padding:
            train_y = np.array(train_y).reshape(1,4,4,4)
                
        elif "same" in self.padding:
            train_y = np.array(train_y).reshape(1,6,6,8)
                
        elif "back" in self.padding:   
            train_y = np.array(train_y).reshape(1,5,5,8)
            
        file_name = 'input_image.txt'
        self.write_output(file_name, train_x)  
        
        np.save(os.path.join(self.folder_parameter, 'train_x'), train_x)
        
        return train_x, train_y
        
    def add_padding(self, input_data):
        output = np.array([])
        
        padding = list()
        for x in range(input_data.shape[3]):
            padding.append(0.0)
                
        output_shape = 0
        for index, x in enumerate(input_data[0]):
            if self.stride == 1:
                x = np.concatenate(([padding], x))
            x = np.concatenate((x, [padding]))
            output_shape = len(x)
            output = np.append(output, x)
            
        padding = list()
        for x in range(output_shape * input_data.shape[3]):
            padding.append(0.0)
        
        if self.stride == 1:
            output = np.append(padding, output)
            
        output = np.append(output, padding)
        
        return output.reshape(1, output_shape, output_shape, input_data.shape[3])
    
    def valid2(self, train_x):
        if "valid" in self.padding:
            output = train_x
                
        elif "same" in self.padding:
            output = self.add_padding(train_x)
                
        elif "back" in self.padding:   
            output = self.add_padding(train_x)
            
        return output
        
    def valid(self):
        train_x = np.load(os.path.join(self.folder_parameter, 'train_x.npy'))
       
        if "valid" in self.padding:
            output = train_x
                
        elif"same" in  self.padding:
            output = self.add_padding(train_x)
                
        elif "back" in self.padding:   
            output = self.add_padding(train_x)
        
        print(output)
        file_name = 'padding_valid.txt'
        self.write_output(file_name, output)
        
    def model(self):
        shape = (4,4,8)
        input = Input(shape=shape)
        if "valid" in self.padding:
            x = ZeroPadding2D(padding=((0, 0),(0,0)), data_format=None)(input)
        elif "same" in self.padding:
            x = ZeroPadding2D(padding=((1, 1),(1,1)), data_format=None)(input)
        elif "back" in self.padding:   
            x = ZeroPadding2D(padding=((0, 1),(0,1)), data_format=None)(input)
            
        model = Model(inputs=input, outputs=x)
        return model
    
    def output(self, model, data_out):
        file_name = 'padding.txt'
        self.write_output(file_name, data_out)
        
        data_out_binary = self.quantizes_valid.values_to_binary(data_out.reshape(-1, 1))
        print(data_out_binary)
        
        file_name = 'padding'
        self.write_output(file_name, data_out_binary)
        
        print('output data', data_out)
    
    def write_output(self, file_name, values):
        if not os.path.isdir(self.folder_parameter):
                os.makedirs(r'%s/%s' % ('Parameter', 'Padding_' + self.padding))
                
        path_file = os.path.join(self.folder_parameter, file_name)
        
        with open (path_file, 'w') as f:
            f.writelines(str(values))