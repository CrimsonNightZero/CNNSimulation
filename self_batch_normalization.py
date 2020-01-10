# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:21:20 2019

@author: foryou
"""

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from quantize_valid import QuantizesValid
from read_data import ReadData
import os
from keras import backend as K

class SelfBatchNormalization:
    def __init__(self, mode):
        self.quantizes_valid = QuantizesValid()
        self.read_data = ReadData()
        self.mode = mode
        self.folder_parameter = os.path.join('Parameter', self.mode)
        
    def data(self):
        train_x = list(np.arange(4,-4,-0.125)[0:64])
        train_x.extend(np.arange(-4,4,0.125)[0:64])
        
        train_y = list(np.arange(8,-8,-0.125)[0:64])
        train_y.extend(np.arange(-8,8,0.125)[0:64])
        
        train_x_binary = self.quantizes_valid.values_to_binary(train_x)
        file_name = 'input_image'
        self.write_output(file_name, train_x_binary)
    
        train_x = self.quantizes_valid.quantizes(train_x)
        train_x = np.array(train_x).reshape(1,4,4,8)
        
        train_y = self.quantizes_valid.quantizes(train_y)
        train_y = np.array(train_y).reshape(1,4,4,8)
        
        file_name = 'input_image.txt'
        self.write_output(file_name, train_x) 
        
        np.save(os.path.join(self.folder_parameter, 'train_x'), train_x)
        
        return train_x, train_y
        
    def valid2(self, train_x, output_shape, parameter):
        weights = list()
        for index in range(len(parameter.values())):
            weight = list()
            weight.append(parameter['gamma'][index])
            weight.append(parameter['beta'][index])
            weight.append(parameter['mean'][index])
            weight.append(parameter['variance'][index])
            
            a = self.quantizes(weight)
            weights.append(a)
            
        print('quantizes weight:', weights)
        
        for index, x in enumerate(train_x.reshape(-1, 1)):
            a = weights[index % train_x.shape[3]]
            if self.mode == "batchnorm1" or self.mode == "batchnorm2":
                x = (x - a[2])  * a[3] * a[0] + a[1]
            elif self.mode == "batchnorm3":
                x = x * a[3] + a[2]  
                
    def valid(self):
        train_x = np.load(os.path.join(self.folder_parameter, 'train_x.npy'))
        
        parameter = {'gamma':list(), 'beta':list(), 'mean':list(), 'variance':list()}
        
        for weight in parameter.keys():
            self.read_data.file = os.path.join(self.folder_parameter, weight + '.txt')
            parameter[weight] = self.read_data.read_values()
        
        weights = list()
        for index in range(len(parameter.values())):
            weight = list()
            weight.append(parameter['gamma'][index])
            weight.append(parameter['beta'][index])
            weight.append(parameter['mean'][index])
            weight.append(parameter['variance'][index])
            
            a = self.quantizes(weight)
            weights.append(a)
            
        print('quantizes weight:', weights)
        
        data_out = ''
        for index, x in enumerate(train_x.reshape(-1, 1)):
            a = weights[index % train_x.shape[1]]
            if "batchnorm3" in self.mode or "batchnorm2" in self.mode:
    #            print(x - a[2])
    #            print((x - a[2])* a[3])
    #            print((x - a[2])* a[3] * a[0])
    #            print((x - a[2])* a[3] * a[0] + a[1])
                x = (x - a[2])  * a[3] * a[0] + a[1]
            elif "batchnorm3" in self.mode:
#                print(x)
#                print(x * a[3]) 
#                print(x * a[3] + a[2] )
                x = x * a[3] + a[2]  
                
            data_out += str(x)+ '\n'
            print('output data:', x)
        file_name = 'batchnorm_valid.txt'
        self.write_output(file_name, data_out)
            
    def model(self):
        shape = (4,4,8)
        input = Input(shape=shape)
        x = BatchNormalization()(input)
        model = Model(inputs=input, outputs=x)
        return model
    
    def quantizes(self, weights):
        if "batchnorm1" in self.mode:
            weights[3] =  1 / np.sqrt(weights[3])
            weights = self.quantizes_valid.quantizes(weights)
            
        elif "batchnorm2" in self.mode:
            weights[3] =  np.sqrt(weights[3])
            for index, x in enumerate(self.quantizes_valid.quantizes(weights)):
                weights[index] = x
                
            weights[3] = 1 / weights[3]
            weights[3] = self.quantizes_valid.quantizes([weights[3]])
        elif "batchnorm3" in self.mode:
            weights[2] = - (1 / np.sqrt(weights[3]) * weights[2] * weights[0]) + weights[1]
            weights[3] = 1 / np.sqrt(weights[3]) * weights[0]
            weights = self.quantizes_valid.quantizes(weights)
            
        return weights
    
    def training(self, model):
        files_name = ['gamma', 'beta', 'mean', 'variance']
        variance = list()
        mean = list()
        gamma = list()
        beta = list()
        
        
        for index in range(len(files_name)):
            weight_output = ""
            for x in model.layers[1].get_weights()[index]:
                weight_output += str(x) + '\n'
                
            file_name = files_name[index] + '.txt'
            self.write_output(file_name, weight_output)
            print('original weight :', files_name[index])   
            
        for index in range(len(model.layers[1].get_weights()[0])):
            weights = list()
            weights.append(model.layers[1].get_weights()[0][index])
            weights.append(model.layers[1].get_weights()[1][index])
            weights.append(model.layers[1].get_weights()[2][index])
            weights.append(model.layers[1].get_weights()[3][index])
            print('original weight :', weights)   
                
            weights = self.quantizes(weights)
            variance.append(weights[3])
            mean.append(weights[2])
            gamma.append(weights[0])
            beta.append(weights[1])
            
        variance = self.quantizes_valid.values_to_binary(variance)
        file_name = 'variance'
        self.write_output(file_name, variance)
        
        mean = self.quantizes_valid.values_to_binary(mean)
        file_name = 'mean'
        self.write_output(file_name, mean)
        
        gamma = self.quantizes_valid.values_to_binary(gamma)
        file_name = 'gamma'
        self.write_output(file_name, gamma)
        
        beta = self.quantizes_valid.values_to_binary(beta)
        file_name = 'beta'
        self.write_output(file_name, beta)

    #    weights = quantizes(weights)
        
    #    weights_quantize = list()
    #    for x in weights:
    #        weights_quantize.append(np.array([x], dtype = np.float32))
    #    
    #    print('quantizes weight :', weights_quantize)
    #    model.layers[1].set_weights(weights_quantize)
        print(model.layers[1].get_weights())
    #    
        json_string = model.to_json()
        with open(os.path.join(self.folder_parameter, "batchnorm model.json"), "w") as text_file:
            text_file.write(json_string)
        model.save(os.path.join(self.folder_parameter, "batchnorm model.hdf5"))
        return model
    
    def output(self, model, data_out):
        file_name = 'batchnorm.txt'
        self.write_output(file_name, data_out.reshape(-1, 1))
        
        data_out_binary = self.quantizes_valid.values_to_binary(data_out.reshape(-1, 1))
        print(data_out_binary)
        file_name = 'batchnorm'
        self.write_output(file_name, data_out_binary)
        
        print('output data', data_out)
        print('gamma:', K.eval(model.layers[1].gamma))
        print('beta:', K.eval(model.layers[1].beta))
        print('moving_mean:', K.eval(model.layers[1].moving_mean))
        print('moving_variance:', K.eval(model.layers[1].moving_variance))
#        print('epsilon :', model.layers[1].epsilon)
#        print('data_in :', data_in)
#        print('data_out:', data_out)
        
    def write_output(self, file_name, values):
        if not os.path.isdir(self.folder_parameter):
            os.makedirs(r'%s/%s' % ('Parameter', self.mode))
            
        path_file = os.path.join(self.folder_parameter, file_name)
            
        with open (path_file, 'w') as f:
            f.writelines(str(values))