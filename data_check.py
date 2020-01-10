# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 13:35:44 2019

@author: foryou
"""

import numpy as np
from read_data import ReadData
from quantize_valid import QuantizesValid
import os

def get_folder():
    if file_name == 'batchnorm':
        folder = mode
    elif file_name == 'dense':
        folder = 'Dense_channel512'
    elif file_name == 'globalaverage':
        folder = 'Globalaverage'
    elif file_name == 'convolution':
        folder = 'Convolution_3v3_channel8_valid_stride1'
    elif file_name == 'padding':
        folder = 'Padding_back_channel8'
    return folder
#Convolution_3v3_channel8_valid_stride1
#Convolution_3v3_channel12_same_stride1
#Convolution_1v1_same_stride1
#Convolution_3v3_back_stride2
#Convolution_3v3_same_stride1
#Convolution_3v3_same_stride2
#Convolution_3v3_valid_stride1
#Convolution_3v3_valid_stride2
#DepthwiseConv_3v3_back_stride2
#DepthwiseConv_3v3_same_stride1
#DepthwiseConv_3v3_same_stride2
#DepthwiseConv_3v3_valid_stride1
#DepthwiseConv_3v3_valid_stride2
    
#batchnorm dense convolution globalaverage mobilenetv2cifar10 padding
file_name = "mobilenetv2cifar10"
#batchnorm1 2 3
mode = "batchnorm3"

#folder = get_folder()
#Mobilenetv2Cifar10 Conv1 DwConv bottleneck0_layer bottleneck1_layer
#bottleneck2_layer Conv_3v3
folder_parameter = os.path.join('Parameter', "Mobilenetv2Cifar10")
#folder_parameter = os.path.join('Parameter', folder)
#software hardware
selector = "hardware"

if __name__ == '__main__': 
    valid = ''
    read_data = ReadData()
    quantize_valid = QuantizesValid()
    if selector == "hardware":
        read_data.type = 'binaries'
        read_data.file = os.path.join(folder_parameter, 'output_verilog.txt')
        binaries = read_data.read_values()
        print(binaries)
#        binaries = quantize_valid.binary_to_values(binaries)
        binaries = quantize_valid.binary_to_values(binaries[0:10])
        valid = binaries
        print(binaries)
    elif selector == "software":
        read_data.file = os.path.join(folder_parameter, file_name + '_valid.txt')
        values = read_data.read_values()
        valid = values
        print(values)
        
    print("aab", file_name == "mobilenetv2cifar10")
    if file_name == "mobilenetv2cifar10":
        #read_data.type = 'binaries'
        #read_data.file = os.path.join(folder_parameter, file_name)
#        source = read_data.read_values()
        #source = quantize_valid.binary_to_values(source)
        read_data.type = 'values'
        read_data.file = os.path.join(folder_parameter, file_name + '.txt')
        source = read_data.read_values2()
    else:
        read_data.type = 'values'
        read_data.file = os.path.join(folder_parameter, file_name + '.txt')
        source = read_data.read_values()
    
    print("aaa")
    print(source)
#    print(valid)
    total = 0
    values = ""
    error = ""
    for x in range(len(source)):
        print(x, np.float32(abs(abs(source[x]) - abs(valid[x]))))
        total += np.float32(abs(abs(source[x]) - abs(valid[x]))) / len(source)
        values += str(source[x]) + " " + str(valid[x]) + " " + str(np.float32(abs(abs(source[x]) - abs(valid[x])))) + " " + str(total) + "\n"
        if np.float32(abs(abs(source[x]) - abs(valid[x]))) > 1:
            error += str(x) + " " + str(source[x]) + " " + str(valid[x]) + " " + str(np.float32(abs(abs(source[x]) - abs(valid[x])))) + " " + str(total) + "\n"
    
    file_name = os.path.join(folder_parameter, 'comparetable.txt')
    with open (file_name, 'w') as f:
        f.writelines(str(values))
        
    file_name = os.path.join(folder_parameter, 'error.txt')
    with open (file_name, 'w') as f:
        f.writelines(str(error))
        
    print(np.float32(total))