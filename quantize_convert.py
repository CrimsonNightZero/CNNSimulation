# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:21:02 2018

@author: foryou
"""

import numpy as np 
import struct
import ctypes
from sklearn import cluster
import tensorflow as tf
import os
from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

class QuantizeFloat:
    def __init__(self, bit, bitW):
        self.bit = bit
        self.bitW = bitW
    
    def quantize(self, x, k):
        n = float(2 ** k - 1)
    
        return tf.round(x * n) / n

    def fw(self, x,  bit= None):
        if bit is None:
            bit = self.bit
            
        if bit == 32:
            return x
    
        if bit == 1:   # BWN
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
    
            return tf.sign(x / E) * E
    
        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * self.quantize(x, bit) -1
    
    def float_to_cutbit(self, x_group, bit = None):
        x_flat = x_group.reshape(-1,1)
        if bit is None:
            bit = self.bit
        bit = pow(2, bit)
        x_len = len(x_flat)
        x_cut = int((x_len - 1) / (bit - 1))
        x_remainder = int(abs(x_len - x_cut * (bit - 1)) / 2)
#        print(x_len)
#        print(x_cut)
#        print(x_remainder)
#        print(x_group.reshape(-1,1))
        x = np.sort(x_group, axis = None)
#        print(x)
        cut_value = list()
        for value in range(bit):
#            print(x_cut*cut+x_low)
            cut_value.append(x[x_cut * value + x_remainder])
#        print(cut_value)
        cut_range = list()
        for index, value in enumerate(cut_value):
            if index == 0:
                cut_range.append(value)
            else:
                cut_range.append((cut_value[index - 1] + cut_value[index]) / 2) 
        cut_range.append(cut_value[-1])
#        print(cut_range)
        
        for x_index, x in enumerate(x_flat):
            for cut_index, value in enumerate(cut_value):
                if cut_range[cut_index] <= x < cut_range[cut_index + 1]:
                    x_flat[x_index] = value
                    break
            if x >= cut_value[-1]:
                x_flat[x_index] = cut_value[-1]
            elif x < cut_value[0]:
                x_flat[x_index] = cut_value[0]
            
        return x_flat.reshape(x_group.shape)
    
    def float_to_binary2(self, x):
#        if x >= 1.0 or x <= -1.0:
#            exp = x * pow(2, self.bit) - (x / abs(x))
#            binary = "{0:b}".format(int(exp)) 
#            return binary
#        print(x)
        exp = x * pow(2, self.bit)
#        print(exp)
#        binary = "{0:b}".format(int(exp)).replace('-', '')
#        if binary == '0000':
#            binary = '0001'
#        print(binary)
#        print(binary)
#        binary_len = len(binary)
#        if  binary_len < self.bit:
#            binary = (self.bit - binary_len) * '0' + binary
    #    binary = bin(int(exp))
            
#        if float(x) < 0:
#            return '-' + str(binary)
        
        return exp
    def binary(self, value):
        sign = np.sign(value)
        modf, integer = np.modf(np.abs(value))
#        bit = 32
#        modf = modf * pow(2, bit)
    #            print(np.round(np.log2(modf)))
        modf = np.power(2,np.round(np.log2(modf)))
        modf, modf_integer = np.modf(modf)
        integer = np.add(integer, modf_integer)
        integer = np.power(2, np.round(np.log2(integer)))
        dic2 = np.add(integer, modf)
        dic2 = np.multiply(sign, dic2, dtype= np.float32)
        return dic2
    
    def float_to_binary(self, x, bit = None):
        if bit is None:
            bit = self.bit
#        print(bit)
        if x >= 1.0 or x <= -1.0:
#            print(x * pow(2, bit))
            exp = x * pow(2, bit) - (x / abs(x))
            binary = "{0:b}".format(int(exp)) 
            return binary
#        print(x)
        exp = x * pow(2, bit)
#        print(exp)
#        binary = self.binary(exp)
#        print(binary)
        binary = "{0:b}".format(int(exp)).replace('-', '')
#        if binary == '0000':
#            binary = '0001'
#        print(binary)
        binary_len = len(binary)
        if  binary_len < bit:
            binary = (bit - binary_len) * '0' + binary
    #    binary = bin(int(exp))
            
        if float(x) < 0:
            return '-' + str(binary)
        
        return binary
       
    def binary_to_float(self, x):
        if x == "":
            return np.float32(0.0)
        exp = 0.0
        index = 1.0
        for value in x.replace('-', ''):
            if int(value) == 1:
                exp += 1.0 / pow(2, index)
#                print(exp)
            index += 1
            
        if float(x) < 0:
            return -np.float32(exp)
        
        return np.float32(exp)
    
    def float_to_iee754(self, x):
        value = ctypes.c_uint.from_buffer(ctypes.c_float(x)).value
        value = "{0:b}".format(int(value))
        if len(value) < 32:
           value = (32 - len(value)) * '0' + value
        exponent = 1
        mantissa = self.bit-1
#        if int(self.bit % 2) == 0:l
#            mantissa = int(self.bit / 2)
#        else:
#            mantissa = int(self.bit / 2 + 1)
#            
#        exponent = int(self.bit / 2)
         
        
        if exponent < 9:
            iee754 = value[9-exponent:9] + value[9:mantissa+9]
        else:
            iee754 = value[1:9] + value[9:mantissa+9]
         
        if int(value[0]) == 1:
            return '-' + iee754
        return iee754
    
    def float_to_kmeans(self, group_x, bit = None):
        if bit is None:
            bit = self.bit
        group_x = group_x.reshape(-1,1)
        kmeans_fit = cluster.KMeans(n_clusters = pow(2, bit), random_state=0).fit(group_x)
        kmeans_array = kmeans_fit.cluster_centers_[kmeans_fit.labels_]
        cluster_centers = np.array(sorted(kmeans_fit.cluster_centers_))
           
        return kmeans_array, kmeans_fit.labels_, cluster_centers
    
    def float_to_uniform(self, group_x, bit = None):
        if bit is None:
            bit = self.bit
        bit = pow(2, bit)
        x_flat = group_x.reshape(-1,1)
        max_value = max(x_flat)
        min_value = min(x_flat)
        
#        print((max_value + min_value)/bit)
        dics = dict() 
        dics2 = dict() 
        for data in x_flat:
#            print(data)
            for x in range(bit+2):
                if x == 0:
                    continue
                
                if x == 1:
                    splite_min = min_value
                    splite_max = ((max_value + min_value)/bit) * (x + 1)
                elif x == bit+1:
                    splite_min = ((max_value + min_value)/bit) * (x - 1)
                    splite_max = max_value + 1
                else:
                    splite_min = ((max_value + min_value)/bit) * (x - 1)
                    splite_max = ((max_value + min_value)/bit) * x
                    
                if splite_min <= data < splite_max:
                    if not x in dics.keys():
                        dics[x] = list()
                        dics[x].append(data[0])
                        dics2[x] = [data[0] ,1]
                    else:
                        dics[x].append(data[0])
                        dics2[x][0] += data[0] 
                        dics2[x][1] += 1
                    break
        
        labels = dict()   
        for key, values in dics.items():
            for x in values:
                labels[x] = dics2[key][0] / dics2[key][1]
#                labels[x] = np.sum(values) / np.size(values)
#                labels[x] = np.sum(values) / len(values)
                
        for index, data in enumerate(x_flat):
            x_flat[index] = labels[data[0]]
                
        return x_flat.reshape(group_x.shape)
    
    def float_to_hierarchical(self, group_x, bit = None):
        if bit is None:
            bit = self.bit
        group_x = group_x.reshape(-1,1)
        hclust = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n_clusters = bit).fit(group_x)
        
        linked = linkage(group_x)
        
#        plt.figure(figsize=(10, 7))
        label = dendrogram(linked,
                    orientation='top',
                    labels=hclust.labels_,
                    distance_sort='descending',)
        
        value = dendrogram(linked,
                    orientation='top',
                    labels=group_x,
                    distance_sort='descending',)
        
        
        values = dict()
        for label_index, label in enumerate(label['ivl']):
            if not str(label) in values.keys():
                values[str(label)] = list()
                values[str(label)].append(value['ivl'][label_index][0])
            else:
                values[str(label)].append(value['ivl'][label_index][0])
        
        labels = list()
        for x in range(bit) :   
            labels.append(sum(values[str(x)]) / len(values[str(x)]))
#        print(labels)
            
        dics = dict()
        for label, value in values.items():
            for x in value:
                dics[x] = labels[int(label)]
#        print(dics)
        
        hierarchical_array = list()
        for x in group_x:
            hierarchical_array.append(dics[x[0]])
#        print(hierarchical_array)
        del hclust
        return hierarchical_array
    
    def spe_binary(self, value):
        sign = np.sign(value)
        modf, integer = np.modf(np.abs(value))
                
        #            print(np.round(np.log2(modf)))
        modf = np.power(2,np.round(np.log2(modf)))
                
        modf, modf_integer = np.modf(modf)
        integer = np.add(integer, modf_integer)
        integer = np.power(2, np.round(np.log2(integer)))
        dic2 = np.add(integer, modf)
        dic2 = np.multiply(sign, dic2)
        return dic2

    def float_to_linear_transform(self, group_x, bit = None):
        if bit is None:
            bit = self.bit
        bit = pow(2, bit)
#        clusters = 12
#        iris_X = np.array([11.1,12.1,21.2, 44.9, -11.1, 60.5,54.9, -22.8, 54.5, 54.2, 74.1, 80.2]).reshape(-1,1)
        x_flat = group_x.reshape(-1,1)
        max_value = max(x_flat)
        min_value = min(x_flat)
        new_max = max_value / (max_value - min_value)
        new_min = min_value / (max_value - min_value)
#        mean = (new_max + new_min) / 2
#        new_min = self.spe_binary(new_min)
        #print(max_value, min_value)
        #print(new_max, new_min)
        #print(abs(max_value - min_value))
        dics = dict() 
        dics2 = dict() 
            
        #splite = list()
        #for data in iris_X:
        #    splite.append( ((data - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min)
            
        #print(splite)
        #splite_d = list()
        #for x in range(clusters+1):
        #    splite_d.append((new_max - new_min)*x/clusters + new_min)
        
        #print(splite_d)   
        for index , data in enumerate(x_flat):
            for x in range(bit+1):
                if x == 0:
                    continue
        #        print(splite_d[x-1], splite[index], splite_d[x])
                split_min = (new_max - new_min) * (x-1) / bit + new_min
                split_max = (new_max - new_min) * x / bit + new_min
#                split_max = self.spe_binary(split_max)
#                split_min = self.spe_binary(split_min)
                
                
                splite = ((data - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
                
                if split_min <= splite <= split_max:
#                    print('a')
                    if not x in dics.keys():
                        dics[x] = list()
                        dics[x].append(data[0])
                        dics2[x] = [data[0] ,1]
                    else:
                        dics[x].append(data[0])
                        dics2[x][0] += data[0] 
                        dics2[x][1] += 1
                    break
#            print(split_min, split_max)
#            print(split_min <= splite <= split_max)
#            print(data[0])
#        print(dics)
#        print(dics2)
                    
        labels = dict()   
        for key, values in dics.items():
            for x in values:
                labels[x] = dics2[key][0] / dics2[key][1]
#        print(labels)
        for index, data in enumerate(x_flat):
#            print(data)
            x_flat[index] = labels[data[0]]
        return  x_flat.reshape(group_x.shape)

    def int_float_count(self, group_x):
#        group_x = group_x.reshape(-1,1)
        int_max_count = 0
        float_max_count = 0
        sign = True
        int_boolean = False
        for x in group_x:
            if x < 0:
                sign = False
            if x > 1:
                int_boolean = True
            int_count = len(str(int(x)))
            float_count = len(str(x)) - int_count - 1  
            
            int_count = len("{0:b}".format(abs(int(x))))
            if int_max_count < int_count:
                int_max_count = int_count
                
            if  float_max_count < float_count:
                float_max_count = float_count
            
        if not int_boolean:
            int_max_count = 0
        return int_max_count, float_max_count, sign
    
    def int_float_count2(self, group_x):
        int_max_count = 0
        float_max_count = 0
        int_min = 0
#        print(group_x)
        for x in group_x:
            x = abs(x)
            int_count = len(str(x).split('.')[0])
            float_count = len(str(x).split('.')[1])
#            print(x,int_count,float_count)
            if int_max_count < int_count:
                int_max_count = int_count
                
            if  float_max_count < float_count:
                float_max_count = float_count
                
            if int_min < int(str(x).split('.')[0]):
                int_min = int(str(x).split('.')[0])
#                print(int_min)
                
        if int_min == 0:
            int_max_count -= 1
            float_max_count += 1
        return int_max_count, float_max_count        
  
if __name__ == '__main__':
    aa=r'C:\Users\foryou\Desktop\tensorpack-master\mobilenet_cifar10_min_quatize_w\mobilenet_min_dorefa_cifar10-4,32,32\bitW_4_linearTransform_npz\res4_bn1_variance_EMA_linearTransform.npy'
    values = np.load(aa)
#    print(list(data))
#    aaaaa
    binarys = list()
    decimal_point = list()
    quantize_float  = QuantizeFloat(4,4)
#    values = [1.7642739, 0.96490204, 1.0831639, 0.61296976, 2.2943826, 1.7918926, 1.2382102, 1.348595, 1.443095, 0.70464474, 0.64272773, 1.6009408, 1.0236007, 0.9368599]
    int_max_count, float_max_count = quantize_float.int_float_count2(values)
    if int_max_count + float_max_count > 8:
        float_max_count = float_max_count + 8 - (int_max_count + float_max_count)
    print(int_max_count, float_max_count)
   
#                    float_max_count = 0
#                    print(int_max_count, float_max_count)
    for value in values:
        sign = np.sign(value)
#                        print('sign', type(sign))
        modf, integer = np.modf(np.abs(value))
        integer = quantize_float.binary(integer)
#                        print(float_max_count)
        binary = quantize_float.float_to_binary(modf,float_max_count)
        binarys.append(binary)
        modf = quantize_float.binary_to_float(binary)
#                        print('modf', type(modf))
#                        if integer >= 1000:
#                            integer = 512.0
#                        print('integer', type(np.float32(integer)))
        value = (np.float32(integer) + modf) * sign
#                        print('value',type(value))
#                        print((integer + modf), (integer + modf) * sign)
#                        print(type(value))
        decimal_point.append(value)
    print(binarys)
    print(decimal_point)
    
    aaaaaaa
#    quantizeFloat  = QuantizeFloat(2,2)
#    value = -8.8726533e-07
#    print(value)
#    decimal_point = list()
#    binary = quantizeFloat.float_to_binary(value,32)
#    print(binary)
##    binarys.append(binary)
#    decimal_point.append(quantizeFloat.binary_to_float(binary))
#    print(decimal_point)
#    aaaaa
    
    
#    gamma = tf.Variable(tf.ones([out_channels]))
#    beta = tf.Variable(tf.zeros([out_channels]))
#
#    pop_mean = tf.Variable(tf.zeros([out_channels]), trainable=False)
#    pop_variance = tf.Variable(tf.ones([out_channels]), trainable=False)
#    batch_mean, batch_variance = tf.nn.moments(layer, [0, 1, 2], keep_dims=False)
#    epsilon = 1e-3
#
#    aa = tf.nn.batch_normalization([0.1,0.1,0.5,0.1], batch_mean, batch_variance, beta, gamma, epsilon)
#    print(aa)
    conv1_1_bn = tf.layers.batch_normalization(
            inputs=[0.1,0.1,0.5,0.1],
            name='conv1_1_bn'
        )
    print(conv1_1_bn)
    binarys = list()
    decimal_point = list()
    quantizeFloat  = QuantizeFloat(2,2)
    
#    values = [665.039, 715.97955, 1005.9009, 929.59456456406, 657.8306, 548.0846, 761.4505, 639.99524, 683.1145, 518.609, 960.7191, 604.7127, 875.8183, 815.0631, 1255.4917, 801.0748, 1367.0635, 826.8562, 844.362, 736.2404, 748.83325, 729.50275, 751.02124, 731.0314, 850.16644, 924.35754, 591.8801, 823.08466, 1018.4317, 843.0542, 674.55255, 766.54285, 1227.3091, 754.06384, 1028.0327, 907.6794, 787.2757, 804.60895, 882.59937, 637.90497, 781.4935, 780.1573, 1143.489, 840.0016, 1338.1016, 909.3429, 822.55963, 641.7461, 611.0308, 1092.1305, 648.739, 928.1167, 831.688, 770.18475, 885.0575, 822.5212, 869.90686, 724.1308, 697.8953, 832.6207, 840.83, 526.7167, 825.04517, 754.26154, 1216.144, 657.0263, 784.14954, 1063.4851, 601.531, 1108.1217, 879.79895, 833.10364, 628.37067, 955.87714, 814.46747, 943.6724, 676.6947, 821.35956, 799.674, 879.1277, 963.58594, 1013.0311, 912.6822, 1228.6359, 733.7642, 835.0669, 697.8933, 863.6426, 935.4835, 915.8219, 987.03815, 905.5498, 1188.0677, 718.3927, 1282.5127, 992.03925, 1524.0647, 901.9696, 737.21716, 1054.0293, 830.0927, 987.6447, 622.95276, 882.40344, 595.92896, 834.8071, 689.6089, 690.67554, 819.8717, 729.7397, 896.11414, 595.64185, 1005.22266, 701.22504, 1159.214, 708.1707, 740.28516, 843.3157, 888.41376, 488.8902, 921.67084, 770.8093, 1006.84344, 1008.67413, 813.6774, 536.194, -1014.2497, 692.4275]
#    values = [-0.1248431126, -0.5386404, -0.081958376, -0.52407026, -0.28868073, -0.49017292, -0.5104611, -0.3876146, -0.045914523, -0.29729027, -0.056189496, -0.79785144, -0.18408038, -0.09518612, -0.2938853, 0.025590425, -0.27267256, -0.34640443, -0.15144864, -0.8238847, -0.099755794, -0.41038057, -0.0951613, -0.24942112, -0.21387386, -0.38894704, -0.42539948, -0.2907505, -0.25655583, -0.15318964, -0.467906, -0.14169312, -0.28158334, 0.0009861126, -0.21454169, -0.27338302, -0.2213965, -0.14198922, -0.11004386, -0.20569386, -0.20693755, -0.14251998, -0.26740032, -0.06924267, -0.38218, -0.4110391, -0.2559493, -0.10342196, -0.14456531, -0.5001545, -0.0908284, -0.29633018, -0.18938376, -0.5824517, -0.16263494, -0.22804898, -0.26979423, -0.14977886, -0.29231334, -0.2295512, -0.21191826, -0.25152594, -0.108222134, -0.18823424, -0.089842565, -0.4457327, -0.2784787, -0.030219354, -0.15762767, -0.20681418, -0.015702108, -0.22395381, -0.3881082, -0.22553405, -0.19445574, -0.13193771, -0.111281194, -0.310138, -0.15584865, -0.11899733, -0.16262403, -0.18359631, -0.36201036, -0.1106517, -0.35552314, -0.10339379, -0.13977659, -0.37337378, -0.18671355, -0.39212403, -0.03046997, -0.009870089, 0.02903364, -0.23398234, -0.27318782, -0.31189072, 0.0040608793, -0.13337415, -0.09094333, -0.25589672, -0.21623945, -0.71583724, -0.16714588, -0.4760462, -0.1710165, -0.12214823, -0.13383082, -0.7092574, -0.26059052, -0.13521564, -0.21292728, -0.8541509, -0.17819448, -0.061696704, -0.32579646, -0.086905725, -0.074629806, -0.26521716, -0.1080212, -0.504897, -0.3025241, -0.17147382, -0.2429309, -0.16645235, -0.436393, -0.31487504, -0.11795857, -0.30888614]
#    values = [124.89383, 130.74484, 172.03859, 174.18309, 171.60655, 106.5257, 125.01873, 131.82538, 192.0956, 149.96422, 133.66016, 156.09377, 136.27415, 143.49162, 107.62132, 152.26527, 118.850334, 100.257164, 144.1541, 128.84357, 81.23214, 112.015045, 84.29825, 104.089745, 89.75031, 148.2264, 131.48796, 91.30819, 117.91457, 158.8656, 97.80301, 98.94293, 99.28926, 89.59427, 138.50278, 115.35419]
    values = list()
    for x in range(9):
        value = (pow(2,x)+0.12345678)
        values.append(value)
        
    index = 0
    for x in values:
        values[index] = abs(x)
        index += 1
    int_count, float_max_count = quantizeFloat.int_float_count2(values)
#    int_count += 2
#    print(sign)
    print('int', int_count, 'float32', float_max_count)
#    print(values)
#    if int_count + float_max_count > 8:
#        float_max_count = float_max_count + 8 - (int_count + float_max_count)
    
        
#    float_max_count = float_max_count +1
#    if int_max_count > 0:
#        if sign:
#            float_max_count = float_max_count + (15 - (float_max_count + int_max_count))
#        else:
#            float_max_count = float_max_count + (15 - (float_max_count + int_max_count + 1))
#    if float_max_count >= 8:
#        if sign:
#            float_max_count = float_max_count + (8 - (float_max_count + int_max_count))
#        else:
#            float_max_count = float_max_count + (8 - (float_max_count + int_max_count + 1))
#    float_max_count = 4
#    int_count = int_count +2
#    float_max_count + int_count + 1 >= 15
#    float_max_count >= 8
    print('int', int_count, 'float32', float_max_count)
    
    type_list = dict()
    for value in values:
        sign = np.sign(value)
        
        
        modf, integer = np.modf(np.abs(value))
        integer = quantizeFloat.binary(integer)
        binary = quantizeFloat.float_to_binary(modf, float_max_count)
        binarys.append(binary)
        modf = quantizeFloat.binary_to_float(binary)
#        print(modf)
        value = (integer + modf) * sign
        print(type(value))
#        print(integer)
#        print(integer + modf)
#        value = np.float32(value)
#        value = np.around(value, 10)
#        print((integer + modf), (integer + modf) * sign)
#        print(value)
#        print(type(value))
        decimal_point.append(value)
        
        if not type(value) in type_list.keys():
            type_list[type(value)] = 0
        else:
             type_list[type(value)] += 1
    
    print(binarys)
    print(decimal_point)
    print(type_list)
    print(int_count, float_max_count)
    aaaaaa
    binarys = list()
    decimal_point = list()
    
    sign = np.sign(value)
    modf, integer = np.modf(np.abs(value))
    
    integer = quantizeFloat.binary(integer)
    index = 1
    print(str(value))
    print(str(value))
    print(len(str(value)))
#    print(binary)
#    while(True): 
#        if index == 10:
#            break
#        modf = modf *pow(2, 16)
#        binary = "{0:b}".format(int(modf)).replace('-', '')
#        print(binary)
#        index += 1
#        print(modf)
    print(np.binary_repr(value, width=16))
    print(modf)
    aaaaa
    print(integer)
    
    binary = quantizeFloat.float_to_binary(modf, 16)
    binarys.append(binary)
    modf = quantizeFloat.binary_to_float(binary)
    print(modf)
    value = (integer + modf) * sign
    decimal_point.append(value)
    print(binarys)
    print(decimal_point)
    
    aaaaaa
#    binarys = binaryFloat.float_to_iee754(1.16783567e-02)
#    print(binarys)
#    binarys = binaryFloat.binary_to_float(binarys)
#    print(binarys)
    array =np.array([[0.2,0.3,0.1,0.5], [2.4,32,1.1,0.5], [3.1,2.2,1.5,1.5]])
    print(array)
#    aa = [-0.22790756e-02,-0.22790756e-02,-0.22790756e-02, -2.16783567e-02,  8.42484196e-03-1,92046836e-02, -1.22790756e-02, -1.16783567e-02,  9.42484196e-03]
    bb=quantizeFloat.float_to_cutbit(array)
    print(bb)
#    with tf.Session() as sess:
#                    dic2 = sess.run(quantize_float.fw(dic2))
#        print(sess.run(quantizeFloat.fw(aa)))
  
#    for a in aa:
#        bb=quantizeFloat.float_to_binary(a)
#        print(bb)
#    binary_float = BinaryFloat(4)
#    array = list()
#    for value in range(21):
#        value *= -0.05
#        array.append(value)
#    del array[0]
#    print(array)
#    
#    exp = list()
#    for value  in array: 
#        exp.append(binary_float.float_to_binary(value))
#        
#    print(exp)
#    
#    binary_value = list()
#    for value  in exp: 
#        binary_value.append(binary_float.binary_to_float(str(value)))
#        
#    print(binary_value)
        
#    binary = "{0:b}".format(int(-7864))
#    print(type(-int('111')))
#    print(binary)
#    exp = binary_to_float(binary)
#    print(exp)
    
#    print("{0:.1f}".format(binary)) 
#    print(bin(7864))
    
#    result = binary_float.float_to_binary(-0.25)    
#    print(result)
#    path = r'C:\Users\foryou\Desktop\tensorpack-master\lenet_min_dorefa_mnist-4,4,4\bitW_4_npz\conv0_W.npy'
#    print(np.load(path))
#    path = r'C:\Users\foryou\Desktop\tensorpack-master\lenet_min_dorefa_mnist-4,4,4\bitW_4_binary_npz\conv0_W_binary.npy'
#    print(np.load(path))
#    path = r'C:\Users\foryou\Desktop\tensorpack-master\lenet_min_dorefa_mnist-4,4,4\bitW_4_binary_book_npz\conv0_W_binary_book.npy'
#    print(np.load(path))
#    path = r'C:\Users\foryou\Desktop\tensorpack-master\alexnet_dorefa_mnist-8,8,8\model-4680.npz'
#    data = np.load(path)
#    print(data['fc1/W:0'])
#    print(list(data)
#    path_w = r'C:\Users\foryou\Desktop\tensorpack-master\lenet_mnist_min_quatize_w\lenet_min_dorefa_mnist-2,32,32'
    
    path_w = r'C:\Users\foryou\Desktop\tensorpack-master\lenet_fashionmnist_min_quatize_w\lenet_min_dorefa_fashionmnist-2,32,32'
    path_w_b = r'C:\Users\foryou\Desktop\tensorpack-master\lennet_min_quatize_w_b'
    path_w_v = r'C:\Users\foryou\Desktop\tensorpack-master\lennet_min_quatize_w_v'
    
    paths = dict()
    for dirPath, dirs, files in os.walk(path_w):
        for file in files:
            if 'npz' in file:
                paths[dirPath] = file
#    print(paths)            
    for path, name in paths.items():
#        print(path)
        
        
#        if 'bn1_gamma_dorefa' in name:
#        if 'bitW_2_decimal_point' in path or 'bitW_2_binary' in path or 'bitW_2_dorefa' in path:
        if 'bitW_2_linearTransform_binary' in path:
            data = np.load(os.path.join(path, name))
            print(list(data))
            data = data['bn1/variance/EMA:0']
            print(data)
        elif 'bitW_2_linearTransform_decimal_point' in path or 'bitW_2_linearTransform' in path:  #and 'bitW_2_binary' in name:
            print(path)
            
            data = np.load(os.path.join(path, name))
#            print(data)
            print(list(data))
            data = data['bn1/variance/EMA:0']
            for x in data:
                print('%.32f' % x)
#            break
    aaaaaaaaa  
    path = r'C:\Users\foryou\Desktop\tensorpack-master\lennet_min_quatize_w_v\lenet_min_dorefa_mnist-2,2,2\bitW_2_uniform_npz\bitW_2_uniform.npz'
    data = np.load(path)
    print(list(data)) 
#    for x in list(data):
#        print(data[x])
    aa=data['conv0/W:0']
#    print(data['conv1/W:0'])
    print(aa)
    aaaaaaaaaaaa
    path = r'C:\Users\foryou\Desktop\tensorpack-master\lennet_min_quatize_w_v\lenet_min_dorefa_mnist-4,4,4\bitW_4_dorefa_npz\bitW_4_dorefa.npz'
    data = np.load(path)
    print(list(data)) 
    aaz=data['conv0/b:0']
    print(aaz)
    for a in aaz:
        bb=quantizeFloat.float_to_binary(a)
        print(bb)
    
    aaaaaaaaaa
    path = r'C:\Users\foryou\Desktop\tensorpack-master\lennet_min_quatize_w\lenet_min_dorefa_mnist-4,4,4\bitW_4_kmeans_npz\bitW_4_kmeans.npz'
    data = np.load(path)
    print(list(data)) 
#    for x in list(data):
#        print(data[x])
    a=data['conv0/W:0']
#    print(data['conv1/W:0'])
    print(a)
    
    path = r'C:\Users\foryou\Desktop\tensorpack-master\lennet_min_quatize_w\lenet_min_dorefa_mnist-4,4,4\bitW_4_binary_npz\bitW_4_binary.npz'
    data = np.load(path)
    print(list(data)) 
    b=data['conv0/W:0']
    print(b)
#    print(data['conv1/W:0'])
    
    path = r'C:\Users\foryou\Desktop\tensorpack-master\lennet_min_quatize_w_v\lenet_min_dorefa_mnist-4,4,4\bitW_4_decimal_point_npz\bitW_4_decimal_point.npz'
    data = np.load(path)
    print(list(data)) 
    c=data['conv0/b:0']
    print(c)
#    print(data['conv1/W:0'])
    
    
    path = r'C:\Users\foryou\Desktop\tensorpack-master\lenet_min_dorefa_mnist-4,4,4\lenet_model-4680.npz'
    data = np.load(path)
    print(list(data)) 
#    print(data['conv1/W:0'])
    d=data['conv1/W:0']
#    print(float("101",0.5))
#    print(bin(0.5))
#    print("{0:b}".format(0.5))