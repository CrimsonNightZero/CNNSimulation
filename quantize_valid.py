# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:38:17 2019

@author: foryou
"""

import numpy as np
from quantize_convert import QuantizeFloat

class QuantizesValid:
    def quantizes(self, values):
        bit = 8
        quantize_float = QuantizeFloat(bit,bit)           
        temp =list(values) #object release
        
        int_max_count, float_max_count = quantize_float.int_float_count2(temp)
        if int_max_count + float_max_count > bit:
            float_max_count = float_max_count + bit - (int_max_count + float_max_count)
        print(int_max_count, float_max_count)
#        print(list(temp))
        binarys = list()
        decimal_point = list()
#        print(list(values))
        for value in temp:
#            print(value)
            sign = np.sign(value)
    #                        print('sign', type(sign))
            modf, integer = np.modf(np.abs(value))
    #        integer = quantize_float.binary(integer)
    #                        print(float_max_count)
            binary = quantize_float.float_to_binary(modf,float_max_count)
            
            binarys.append(binary)
            modf = quantize_float.binary_to_float(binary)
    #                        print('modf', type(modf))
    #                        if integer >= 1000:
    #                            integer = 512.0
    #                        print('integer', type(np.float32(integer)))
            value = (np.float32(integer) + modf) * sign
    #        value = (integer + modf) * sign
    #        print(binary,value)
    #                        print('value',type(value))
    #                        print((integer + modf), (integer + modf) * sign)
    #                        print(type(value))
            
    #        value = np.array([value], dtype=np.float32)
            decimal_point.append(value)
        return np.array(decimal_point, dtype=np.float32)
    #    return decimal_point
        
    def values_to_binary(self, values):
        brinaries = ''
#        print(list(values))
        for x in values:
            sign = np.sign(x)
            modf, integer = np.modf(np.abs(x))
        #        modf = int(str(modf).split('.')[1])
            binary_integer = "{0:b}".format(int(integer))
            bit = 7
            binary_len = len(binary_integer)
#            print("binary_integer",binary_integer)
            if  binary_len < bit:
                binary_integer = (bit - binary_len) * '0' + binary_integer
            elif  binary_len > bit:
                binary_integer = binary_integer[binary_len-bit:binary_len]
#            print("binary_integer",binary_integer)
            quantize_float = QuantizeFloat(bit,bit)   
            binary_modf = quantize_float.float_to_binary(modf)
#            print("binary_modf",binary_modf)
            if self.binary_to_int(binary_integer + binary_modf) == 0:
                sign = 1
            binary_output = ''
            if sign < 0:
                for y in binary_integer + binary_modf:
                    if y == '0':
                        binary_output += '1'
                    else:
                        binary_output += '0'
#                print(binary_output)
#                print(self.binary_to_int(binary_output))
                binary_output = "{0:b}".format(self.binary_to_int(binary_output) + 1)
#                print("{0:b}".format(self.binary_to_int(binary_output) + 1))
                brinaries += "1" + binary_output + '\n'
            else:
                binary_output = binary_integer + binary_modf
                brinaries += "0" + binary_output + '\n'
#            print(brinaries)
    #        aaa
    #        print(binary_output, modf, sign, x)
            
        return brinaries
    
    def binary_to_values(self, binaries):
        binary = ''
        values = list()
        quantize_float = QuantizeFloat(7,7) 
        for x in binaries:
    #        print(x,x[0])
            if x[0] == '1': 
                binary = "{0:b}".format(self.binary_to_int(x) - 1)
                binary_output = ''
                for y in binary:
                    if y == '0':
                        binary_output += '1'
                    else:
                        binary_output += '0'
            else:
                binary_output = x
                
#            print(binary_output)
    #        print(binary_to_int(binary_output[1:8]), binary_output[1:8])
    #        print(quantize_float.binary_to_float(binary_output[8:15]), binary_output[8:15])
            
            values_output = self.binary_to_int(binary_output[1:8]) + quantize_float.binary_to_float(binary_output[8:15])
            if x[0] == '1': 
                values_output = values_output * -1
                
            values.append(values_output)
        return values
            
    def binary_to_int(self, binary):
        value = 0
        for index, x in enumerate(reversed(binary)):
            value  += int(x) * (2 ** index)
    #        print(value, index, 2 ** index)
        return value
        
    def twos_comp(self, val, bits):
        """compute the 2's complement of int value val"""
        if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)        # compute negative value
        return val 
    