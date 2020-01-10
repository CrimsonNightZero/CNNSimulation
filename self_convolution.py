# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:12:55 2019

@author: foryou
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, ZeroPadding2D
from quantize_valid import QuantizesValid
from self_padding import SelfPadding
from read_data import ReadData
import os

class SelfConvolution :
    def __init__(self, mode, padding, stride):
        self.quantizes_valid = QuantizesValid()
        self.self_padding = SelfPadding(padding)
        self.self_padding.stride = stride
        self.read_data = ReadData()
        self.mode = mode
        if "Convolution_1v1" in self.mode:
            self.padding = "same"
            self.stride = 1
        elif "back" == padding:
            self.padding = padding
            self.stride = 2
        else:
            self.padding = padding
            self.stride = stride
        self.folder_parameter = os.path.join('Parameter', self.mode + "_" + self.padding + "_stride" + str(self.stride))
        
    def data(self):
        train_x = list(np.arange(8,-8,-0.0625)[0:96])
        train_x.extend(np.arange(-8,8,0.0625)[0:96])
        
              
        if "Convolution_3v3" in self.mode:
            if "valid" == self.padding:
                if 1 == self.stride: 
                    train_y = list(np.arange(8,-8,-0.125)[0:8])
                    train_y.extend(np.arange(-8,8,0.125)[0:8])
                elif 2 == self.stride: 
                    train_y = list(np.arange(8,-8,-0.125)[0:2])
                    train_y.extend(np.arange(-8,8,0.125)[0:2])
            elif "same" == self.padding: 
                if 1 == self.stride:
                    train_y = list(np.arange(10,-10,-0.03125)[0:192])
                    train_y.extend(np.arange(-10,10,0.03125)[0:192])
                elif 2 == self.stride:
                    train_y = list(np.arange(8,-8,-0.125)[0:8])
                    train_y.extend(np.arange(-8,8,0.125)[0:8])
            elif "back" == self.padding: 
                train_y = list(np.arange(8,-8,-0.125)[0:8])
                train_y.extend(np.arange(-8,8,0.125)[0:8]) 
            
        elif "DepthwiseConv_3v3" in self.mode:
            if "valid" == self.padding: 
                if 1 == self.stride: 
                    train_y = list(np.arange(8,-8,-0.125)[0:8])
                    train_y.extend(np.arange(-8,8,0.125)[0:8])
                elif 2 == self.stride: 
                    train_y = list(np.arange(8,-8,-0.125)[0:2])
                    train_y.extend(np.arange(-8,8,0.125)[0:2])
                    
            elif "same" == self.padding: 
                if 1 == self.stride:
                    train_y = list(np.arange(12,-12,-0.125)[0:192])
                    train_y.extend(np.arange(-12,12,0.125)[0:192])
                elif 2 == self.stride:
                    train_y = list(np.arange(8,-8,-0.125)[0:8])
                    train_y.extend(np.arange(-8,8,0.125)[0:8]) 
            elif "back" == self.padding: 
                train_y = list(np.arange(8,-8,-0.125)[0:8])
                train_y.extend(np.arange(-8,8,0.125)[0:8]) 
                
        elif "Convolution_1v1" in self.mode:
           train_y = list(np.arange(8,-8,-0.125)[0:64])
           train_y.extend(np.arange(-8,8,0.125)[0:64])
                    
        train_x_binary = self.quantizes_valid.values_to_binary(train_x)
        file_name = 'input_image'
        self.write_output(file_name, train_x_binary)
        
        train_x = self.quantizes_valid.quantizes(train_x)
#        train_x = np.array(train_x).reshape(1,4,4,4)
        train_x = np.array(train_x).reshape(1,4,4,12)
        
        train_y = self.quantizes_valid.quantizes(train_y)
        if "Convolution_3v3" in self.mode:
            if "valid" == self.padding:
                if 1 == self.stride:
                    train_y = np.array(train_y).reshape(1,2,2,4)
                elif 2 == self.stride:
                    train_y = np.array(train_y).reshape(1,1,1,4)
            elif "same" == self.padding: 
                if 1 == self.stride:
                    train_y = np.array(train_y).reshape(1,4,4,24)
                elif 2 == self.stride:
                    train_y = np.array(train_y).reshape(1,2,2,4)
            elif "back" == self.padding: 
                train_y = np.array(train_y).reshape(1,2,2,4)
                
        elif "DepthwiseConv_3v3" in self.mode:
            if "valid" == self.padding: 
                if 1 == self.stride:
                    train_y = np.array(train_y).reshape(1,2,2,4)
                elif 2 == self.stride:
                    train_y = np.array(train_y).reshape(1,1,1,4)
            elif "same" == self.padding: 
                if 1 == self.stride:
#                    train_y = np.array(train_y).reshape(1,4,4,4)
                    train_y = np.array(train_y).reshape(1,4,4,24)
                elif 2 == self.stride:
                    train_y = np.array(train_y).reshape(1,2,2,4)
            elif "back" == self.padding: 
                train_y = np.array(train_y).reshape(1,2,2,4)
                
        elif "Convolution_1v1" in self.mode:
            train_y = np.array(train_y).reshape(1,4,4,8)
            
        file_name = 'input_image.txt'
        self.write_output(file_name, train_x)
        
        np.save(os.path.join(self.folder_parameter, 'train_x'), train_x)
        np.save(os.path.join(self.folder_parameter, 'train_y'), train_y)
            
        return train_x, train_y
        
    def change_model(self, output, train_x, weights, output_shape):
        #convolution 3*3 DepthwiseConv2D 3*3
        data_test = ""
        if "3v3" in self.mode:
            kernel_size = 3
            if "same" == self.padding or "back" == self.padding:
                train_x = self.self_padding.add_padding(train_x)
                file_name = 'input_image_padding.txt'
                self.write_output(file_name, train_x) 
                train_x_binary = self.quantizes_valid.values_to_binary( train_x.reshape(-1,1))
                file_name = 'input_image_padding'
                self.write_output(file_name, train_x_binary)
                
            print(train_x)
            print(train_x.shape)
            input_width = train_x.shape[2]
            input_data = train_x.reshape(train_x.shape[1] * train_x.shape[2], train_x.shape[3])
                
            y = 0
            channel = 0
            for index, weight in enumerate(weights[0]):
                fliter = index % output_shape[3]
                if "Convolution_3v3" in self.mode:
                    channel_jude = (fliter == 0 and (not index == 0))
                    weight_number = train_x.shape[3] * output_shape[3]
                elif "DepthwiseConv_3v3" in self.mode:
                    channel_jude = (not index == 0)
                    weight_number = train_x.shape[3] * 1
                    
                if channel_jude:
                    if channel < train_x.shape[3] - 1:
                        channel += 1
                    else:
                        channel = 0
                        
                if index / weight_number % kernel_size  == 0 and not y == 0:
                    y += input_width - kernel_size 
                    
                
                width = 0
                col = self.stride
#                int(math.pow(input_width - kernel_size + 1, 2))
                output_size = output_shape[1] * output_shape[2]
                for a in range(output_size):
#                    print(a, channel, fliter, y, y + width, input_data[y + width][channel], weight)
                    data_test += str(a)+","+str(channel)+","+str(fliter)+","+str(y)+","+str(y + width)+","+str(input_data[y + width][channel])+","+str(weight) + '\n'
            
                    output[a][fliter] += input_data[y + width][channel] * weight 
                    if (a+1) % output_shape[1] == 0 and (not a == 0):
                        width = col * input_width
                        col += self.stride
                    else:
                        width += self.stride
#                print(index, channel, fliter, y)
#                print(weight)
    #            output[0][fliter] += input_data[y][channel] * weight 
    #            output[1][fliter] += input_data[y+1][channel] * weight 
    #            output[2][fliter] += input_data[y+2][channel] * weight 
    #            output[3][fliter] += input_data[y+3][channel] * weight
    #            output[4][fliter] += input_data[y+input_width][channel] * weight 
    #            output[5][fliter] += input_data[y+input_width+1][channel] * weight 
    #            output[6][fliter] += input_data[y+input_width+2][channel] * weight 
    #            output[7][fliter] += input_data[y+input_width+3][channel] * weight 
    #            output[8][fliter] += input_data[y+input_width*2][channel] * weight 
    #            output[9][fliter] += input_data[y+input_width*2+1][channel] * weight 
    #            output[10][fliter] += input_data[y+input_width*2+2][channel] * weight 
    #            output[11][fliter] += input_data[y+input_width*2+3][channel] * weight
    #            output[12][fliter] += input_data[y+input_width*3][channel] * weight 
    #            output[13][fliter] += input_data[y+input_width*3+1][channel] * weight 
    #            output[14][fliter] += input_data[y+input_width*3+2][channel] * weight 
    #            output[15][fliter] += input_data[y+input_width*3+3][channel] * weight 
                
                if output_shape[3] == 1:
                    y += 1
                elif fliter == output_shape[3] - 1 and (not index == 0) and channel == train_x.shape[3] - 1:
                    y += 1
            
        elif "Convolution_1v1" in self.mode:
            x_index = 0
            y_index = 0
            for index, x in enumerate(train_x.reshape(-1, 1)):
                for fliter in range(output_shape[3]):
                    if fliter == 0 and y_index == 0 and (not index == 0):
                        x_index += 1
                        
                    print(x_index, fliter, y_index, y_index * output_shape[3] + fliter)
                    print(weights[0][y_index * output_shape[3] + fliter], x)
                    data_test += str(index)+","+str(fliter)+","+str(y_index)+","+str(x_index * output_shape[3] + fliter)+","+str(x)+","+str(weights[0][y_index * output_shape[3] + fliter]) + '\n'
                    output[x_index][fliter] += weights[0][y_index * output_shape[3] + fliter] * x
                    
                    if y_index == train_x.shape[3] - 1 and fliter == output_shape[3] - 1:
                        y_index = 0
                    elif fliter == output_shape[3] - 1:
                        y_index += 1
                        
# DepthwiseConv2D 3*3
#        input_width = train_x.shape[2]
#        input_data = train_x.reshape(train_x.shape[1] * train_x.shape[2], train_x.shape[3])
#        y = 0
#        for index, weight in enumerate(weights[0]):
#            fliter = index % output_shape[3]
#            channel = index % train_x.shape[3]
#            print(weight)
#            print(y % input_width == input_width - 1)
#            if y % input_width == input_width - 1:
#                y += 1
#            
#            print(index,fliter, y, y+1, y+input_width, y+input_width+1)
#            output[0][fliter] += input_data[y][channel] * weight 
#            output[1][fliter] += input_data[y+1][channel] * weight 
#            output[2][fliter] += input_data[y+input_width][channel] * weight 
#            output[3][fliter] += input_data[y+input_width+1][channel] * weight 
##            print(fliter, y, y+1, y+input_width, y+input_width+1)
#            
#            if output_shape[3] == 1:
#                y += 1
#            elif fliter == output_shape[3] - 1 and (not index == 0):
#                y += 1
#        print('output data:', x)
            
        file_name = 'con3v3_test.txt'
        self.write_output(file_name, data_test)
        return output
    
    def valid2(self, train_x, output_shape, parameters):
        weights = list()
        
        parameter = {'weight':list(), 'bias':list()}
        
        weights = list()
        for weight in parameters.reshape(-1,1):
            weights.append(weight[0])
        a = self.quantizes_valid.quantizes(weights)
        weights.append(a)
#        print('quantizes weight:', weights)
        output = list()
        for x in range(output_shape[1] * output_shape[2]):
            temp = list()
            for y in range(output_shape[3]):
                temp.append(0.0)
            output.append(temp)
        print(output)
        
        output = self.change_model(output, train_x, [weights], output_shape)
        return output
    
    def valid(self):
        train_x = np.load(os.path.join(self.folder_parameter, 'Pad.npy'))
        #train_y = np.load(os.path.join(self.folder_parameter, 'train_y.npy'))
        output_shape = (1,16,16,256)#train_y.shape
        
        parameter = {'weight':list(), 'bias':list()}
        
        weights = list()
        for weight in parameter.keys():
           # self.read_data.file = os.path.join(self.folder_parameter, weight + '.txt')
            #parameter[weight] = self.read_data.read_values()
            weight_x = np.load(os.path.join(self.folder_parameter, "conv0_W_dorefa_decimal_point.npy"))
            weight_y = list()
            for x in weight_x.reshape(-1,1):
                weight_y.append(x[0])
            print(weight_y)
            a = self.quantizes_valid.quantizes(weight_y)#parameter[weight])
            weights.append(a)
        print('quantizes weight:', weights)
        
        output = list()
        for x in range(output_shape[1] * output_shape[2]):
            temp = list()
            for y in range(output_shape[3]):
                temp.append(0.0)
            output.append(temp)
        print(output)
        
        output = self.change_model(output, train_x, weights, output_shape)
        
        data_out_view = ''
        data_out = ''
        for index, x in enumerate(output):
            y = index % output_shape[3]
#            x = weights[1][y] + x
#            for index, weight in enumerate(x):
#                if weight < 0:
#                    x[index] = 0.0 
            data_out_view += str(x)+ '\n'
            print('output data:', x)
            for value in x:
                data_out += str(value)+ '\n'
                
        file_name = 'convolution_output.txt'
        self.write_output(file_name, data_out_view)     
        file_name = 'convolution_valid.txt'
        self.write_output(file_name, data_out)
        
    def model(self):
        shape = (4,4,12)
        input = Input(shape=shape)
        if "Convolution_3v3" in self.mode:
            if "back" == self.padding:
                x = ZeroPadding2D(padding=((0, 1),(0,1)), data_format=None)(input)
                x = Conv2D(filters=4,  
                         kernel_size=(3, 3),
                         padding="valid",
                         strides=(2, 2))(x)
            else:
                x = Conv2D(filters=24,  
                         kernel_size=(3, 3),
                         padding=self.padding,
                         strides=(self.stride, self.stride))(input)
            
        elif "DepthwiseConv_3v3" in self.mode:
            if "back" == self.padding:
                x = ZeroPadding2D(padding=((0, 1),(0,1)), data_format=None)(input)
                x = DepthwiseConv2D(  
                         kernel_size=(3, 3),
                         padding="valid",
                         strides=(2, 2))(x)
            else:
                x = DepthwiseConv2D(  
                             kernel_size=(3, 3),
                             padding=self.padding,
                             strides=(self.stride, self.stride))(input)
        elif "Convolution_1v1" in self.mode:
            x = Conv2D(filters=8,  
                     kernel_size=(1, 1),
                     padding='same',
                     strides=(1, 1))(input)
#        activation='relu'
        model = Model(inputs=input, outputs=x)
        return model
    
    def training(self, model):
        files_name = ['weight', 'bias']
        parameter = {'weight':list(), 'bias':list()}
        for index, key in enumerate(parameter.keys()):
            parameter_output = ""
            if "back" == self.padding:
                layer_number = 2
            else:
                layer_number = 1
                
            print(model.layers[layer_number].get_weights()[index])
            
            for x in model.layers[layer_number].get_weights()[index].reshape(-1,1):
                parameter_output += str(x) + '\n'
                parameter[key].append(float(x))
                
            file_name = files_name[index] + '.txt'
            self.write_output(file_name, parameter_output)
            
            weights = self.quantizes_valid.quantizes(parameter[key])
            weights = self.quantizes_valid.values_to_binary(weights)
            self.write_output(files_name[index], weights)
        return model
    
    def output(self, model, data_out):
        file_name = 'convolution.txt'
        self.write_output(file_name, data_out.reshape(-1, 1))
        
        data_out_binary = self.quantizes_valid.values_to_binary(data_out.reshape(-1, 1))
        print(data_out_binary)
        
        file_name = 'convolution'
        self.write_output(file_name, data_out_binary)
        
        print('output data', data_out)
    
    def write_output(self, file_name, values):
        if not os.path.isdir(self.folder_parameter):
            os.makedirs(r'%s/%s' % ('Parameter', self.mode + "_" + self.padding + "_stride" + str(self.stride)))
                
        path_file = os.path.join(self.folder_parameter, file_name)
      
        with open (path_file, 'w') as f:
            f.writelines(str(values))