# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:13:09 2019

@author: foryou
"""

import numpy as np
 
 
def softmax(inputs):
    """
    Calculate the softmax for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))
 
 
softmax_inputs = [-2, -3, -50, -6,1]
print("Softmax Function Output :: {}".format(softmax(softmax_inputs)))