# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:33:44 2019

@author: foryou
"""

class ReadData:
    def __init__(self):
        self.type = 'values'
        self.file = None
        
    def format_form(self, x):
        if self.type == 'values':
            return float(x.strip('\n').replace('[', '').replace(']', ''))
        elif self.type == 'binaries':
            return x.strip('\n').replace('[', '').replace(']', '')
    
    def read_values(self):
        values = list()
        with open(self.file, 'r') as f:
            for x in f.readlines():
                for y in x.split(","):
                    values.append(self.format_form(y))
        return values