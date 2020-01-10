# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:19:02 2018

@author: foryou
"""
from self_average_pooling import SelfAveragePooling
from self_global_average_pooling import SelfGlobalAveragePooling
from self_batch_normalization import SelfBatchNormalization
from self_dense import SelfDense
from self_padding import SelfPadding
from self_convolution import SelfConvolution
from self_mobilenetv2_cifar10 import SelfMobilenetv2Cifar10

#batchnorm1 2 3
mode_batch_normalization = "batchnorm3_channel8"

#Convolution_3v3 DepthwiseConv_3v3 Convolution_1v1
mode_convolution = "Convolution_3v3"
#valid same back
padding_convolution = "valid"
#(1,1) (2,2)
stride_convolution = 2
#Mobilenetv2Cifar10 Conv1 DwConv bottleneck0_layer bottleneck1_layer
#bottleneck2_layer Conv_3v3
mode_mobilenetv2 = "Conv_3v3"

#True False
train_or_valid = False

def Mobilenetv2Cifar10Function():
    self_mobilenetv2_cifar10 = SelfMobilenetv2Cifar10(mode_mobilenetv2)
    if train_or_valid:
        train_x, train_y = self_mobilenetv2_cifar10.data()
        model = self_mobilenetv2_cifar10.model()
        model.compile(loss='mse', optimizer='sgd') 
        model.fit(train_x, train_y, epochs=1)
        print(model.summary())
        model = self_mobilenetv2_cifar10.training(model)
        data_out = model.predict(train_x)
        print(data_out)
        self_mobilenetv2_cifar10.output(model, data_out)
#    else:
#        self_mobilenetv2_cifar10.valid()
        
def ConvolutionFunction():
    self_convolution = SelfConvolution(mode_convolution, 
                                       padding_convolution, 
                                       stride_convolution)
    if train_or_valid:
        train_x, train_y = self_convolution.data()
        model = self_convolution.model()
        model.compile(loss='mse', optimizer='sgd') 
        model.fit(train_x, train_y, epochs=1)
        model = self_convolution.training(model)
        print('input data:', train_x)
        data_out = model.predict(train_x)
        print(data_out)
        self_convolution.output(model, data_out)
    else:
        self_convolution.valid()

def PaddingFunction():
    self_padding = SelfPadding(padding_convolution)
    if train_or_valid:
        train_x, train_y = self_padding.data()
        model = self_padding.model()
        model.compile(loss='mse', optimizer='sgd')
        model.fit(train_x, train_y, epochs=1)
        print('input data:', train_x)
        data_out = model.predict(train_x)
        self_padding.output(model, data_out)
    else:
        self_padding.valid()

def DenseFunction():
    self_dense = SelfDense()
    if train_or_valid:
        train_x, train_y = self_dense.data()
        model = self_dense.model()
        model.compile(loss='mse', optimizer='sgd')
        model.fit(train_x, train_y, epochs=1)
        model = self_dense.training(model)
        print('input data:', train_x)
        data_out = model.predict(train_x)
        print(data_out)
        self_dense.output(model, data_out)
    else:
        self_dense.valid()

def BatchNormalizationFunction():
    self_batch_normalization = SelfBatchNormalization(mode_batch_normalization)
    if train_or_valid:
        train_x, train_y = self_batch_normalization.data()
        model = self_batch_normalization.model()
        model.compile(loss='mse', optimizer='sgd')
        model.fit(train_x, train_y, epochs=1)
        model = self_batch_normalization.training(model)
        print('input data:', train_x)
        data_out = model.predict(train_x)
        print(data_out)
        self_batch_normalization.output(model, data_out)
    else:
        self_batch_normalization.valid()
        
def GlobalAveragePoolingFunction():
    self_global_average_pooling = SelfGlobalAveragePooling()
    if train_or_valid:
        train_x, train_y = self_global_average_pooling.data()
        model = self_global_average_pooling.model()
        model.compile(loss='mse', optimizer='sgd')
        model.fit(train_x, train_y, epochs=1)
        print('input data:', train_x)
        data_out = model.predict(train_x)
        print(data_out)
        self_global_average_pooling.output(model, data_out)
    else:
        self_global_average_pooling.valid()

def AveragePoolingFunction():
    self_average_pooling = SelfAveragePooling()
    train_x, train_y = self_average_pooling.data()
    model = self_average_pooling.model()
    model.compile(loss='mse', optimizer='sgd')
    model.fit(train_x, train_y, epochs=1)
    print('input data:', train_x)
    data_out = model.predict(train_x)
    print(data_out)
   
    
if __name__ == '__main__':
#    Mobilenetv2Cifar10Function()
#    ConvolutionFunction()
    PaddingFunction()
#    DenseFunction()
#    BatchNormalizationFunction()
#    GlobalAveragePoolingFunction()
#    AveragePoolingFunction()
    
#     x = [10.5,21,1,-12,5,-11.0,0.5,-0.4,-1,1]
#     x = tf.tanh(x)
#     
#     with tf.Session() as sess:
#         dic2 = sess.run(x)
#         print(dic2)
#         x = sess.run(x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5)
#         print(x)
#         x = sess.run(2 * quantize(x, 4) -1)
#         print(x)
#    train_x = list(np.arange(200,-200,-0.125)[0:3072])
#    train_y = list(np.arange(8,-8,-0.125)[0:5])
#    train_y.extend(np.arange(-8,8,0.125)[0:5])   
#    from quantize_valid import QuantizesValid
#    quantizes_valid = QuantizesValid()
#    train_x_binary = quantizes_valid.values_to_binary([-127.875])
#    print(train_x_binary)
#    file_name = 'input_image'
#    self.write_output(file_name, train_x_binary)
    
#    weights = [100000.5891,-10000.5]
#    from quantize_valid import QuantizesValid
#    quantizes_valid = QuantizesValid()
#    weights = quantizes_valid.quantizes(weights)
#    print(weights)
#    variance = quantizes_valid.values_to_binary(weights)
#    print(variance)
#    for x in variance.split("\n"):
#        print(len(x))
#    aa="000101111000110"
#    print(len(aa))
