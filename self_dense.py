# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:41:02 2019

@author: foryou
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from quantize_valid import QuantizesValid
from read_data import ReadData
import os

class SelfDense:
    def __init__(self):
        self.quantizes_valid = QuantizesValid()
        self.read_data = ReadData()
        self.folder_parameter = os.path.join('Parameter', 'Dense_channel512')
        
    def data(self):
        train_x = list(np.arange(4,-4,-0.03125)[0:256])
        train_x.extend(np.arange(-4,4,0.03125)[0:256])
        
        train_y = list(np.arange(8,-8,-0.125)[0:5])
        train_y.extend(np.arange(-8,8,0.125)[0:5])
        
        train_x_binary = self.quantizes_valid.values_to_binary(train_x)
        file_name = 'input_image'
        self.write_output(file_name, train_x_binary)
        
        train_x = self.quantizes_valid.quantizes(train_x)
        train_x = np.array(train_x).reshape(1,1,1,512)
             
        train_y = self.quantizes_valid.quantizes(train_y)
        train_y = np.array(train_y).reshape(1,1,1,len(train_y))
            
        file_name = 'input_image.txt'
        self.write_output(file_name, train_x)    
        
        np.save(os.path.join(self.folder_parameter, 'train_x'), train_x)
        np.save(os.path.join(self.folder_parameter, 'train_y'), train_y)
            
        return train_x, train_y
        
    def valid(self):
        train_x = np.load(os.path.join(self.folder_parameter, 'train_x.npy'))
        train_y = np.load(os.path.join(self.folder_parameter, 'train_y.npy'))
        output_shape = train_y.shape
        
        parameter = {'weight':list(), 'bias':list()}
        
        weights = list()
        for weight in parameter.keys():
            self.read_data.file = os.path.join(self.folder_parameter, weight + '.txt')
            parameter[weight] = self.read_data.read_values()
            a = self.quantizes_valid.quantizes(parameter[weight])
            weights.append(a)
        print('quantizes weight:', weights)
        output = list()
        for x in range(output_shape[3]):
            output.append(0.0)
        
#convolution 1*1  
        data_test = ""
        x_index = 0
        y_index = 0
        for index, input_data in enumerate(train_x.reshape(-1, 1)):
            for fliter in range(output_shape[3]):
                if fliter == 0 and y_index == 0 and (not index == 0):
                    x_index += 1
#                if (y_index * output_shape[3] + fliter) == 40:
#                    print(output)
#                    aaaaa  
                print(x_index, fliter, y_index, y_index * output_shape[3] + fliter)
                print(weights[0][y_index * output_shape[3] + fliter], input_data)
                data_test += str(index)+","+str(fliter)+","+str(y_index)+","+str(x_index * output_shape[3] + fliter)+","+str(input_data)+","+str(weights[0][y_index * output_shape[3] + fliter]) + '\n'
                output[x_index * output_shape[3] + fliter] += weights[0][y_index * output_shape[3] + fliter] * input_data
                
                if y_index == train_x.shape[3] - 1 and fliter == output_shape[3] - 1:
                    y_index = 0
                elif fliter == output_shape[3] - 1:
                    y_index += 1
                
#dense
#        for index, x in enumerate(train_x.reshape(-1, 1)):
#            for fliter in range(output_shape[3]):
#                print(weights[0][index * output_shape[3] + fliter], x)
#                output[fliter] += weights[0][index * output_shape[3] + fliter] * x
        file_name = 'dense_test.txt'
        self.write_output(file_name, data_test)        
        print(output)
        data_out = ''
        for index, x in enumerate(output):
            x = weights[1][index] + x
            data_out += str(x)+ '\n'
            print('output data:', x)
            
        file_name = 'dense_valid.txt'
        self.write_output(file_name, data_out)
        
    def model(self):
        shape = (1,1,512)
        input = Input(shape=shape)
        x = Dense(units = 10)(input)
#        x = Dense(units = 10, activation='softmax')(input)
        model = Model(inputs=input, outputs=x)
        return model
    
    def training(self, model):
        files_name = ['weight', 'bias']
        parameter = {'weight':list(), 'bias':list()}
        for index, key in enumerate(parameter.keys()):
            parameter_output = ""
            for x in model.layers[1].get_weights()[index].reshape(-1,1):
                parameter_output += str(x) + '\n'
                parameter[key].append(float(x))
                
            file_name = files_name[index] + '.txt'
            self.write_output(file_name, parameter_output)
            
            weights = self.quantizes_valid.quantizes(parameter[key])
            weights = self.quantizes_valid.values_to_binary(weights)
            self.write_output(files_name[index], weights)
            
        return model
    
    def output(self, model, data_out):
        file_name = 'dense.txt'
        self.write_output(file_name, data_out.reshape(-1, 1))
        
        data_out_binary = self.quantizes_valid.values_to_binary(data_out.reshape(-1, 1))
        print(data_out_binary)
        
        file_name = 'dense'
        self.write_output(file_name, data_out_binary)
        
        print('output data', data_out)
    
    def write_output(self, file_name, values):
        if not os.path.isdir(self.folder_parameter):
            os.makedirs(r'%s/%s' % ('Parameter', 'Dense_channel512'))
                
        path_file = os.path.join(self.folder_parameter, file_name)
      
        with open (path_file, 'w') as f:
            f.writelines(str(values))