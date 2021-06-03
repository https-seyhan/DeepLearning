#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: saul
"""
import numpy as np
import math

# calculate Entropy
class Entropy:   
    def __init__(self, prain=None, pshine = None, p_rain_g_month = None):
        if prain is None:
            self.prain = []  
        elif pshine is None:
            self.pshine = []
        elif p_rain_g_month is None:
            self.p_rain_g_month = []
        self.prain =list(prain /100) 
        self.pshine = list(1 - prain / 100)
        self.p_rain_g_month = np.column_stack([self.prain, self.pshine])
        self.entropyprob = 0.0
        self.indentropies = []
        self.period= 12
       
    def calconditional(self):    
        for i in range(self.period):
            sumentropy = (-1/self.period)*(self.p_rain_g_month[i, 0]*math.log(self.p_rain_g_month[i,0], 2) + self.p_rain_g_month[i,1]*math.log(self.p_rain_g_month[i,1],2))
            self.entropyprob += sumentropy
            self.indentropies.append(sumentropy)
        
                  
entropy = Entropy(np.array([25, 27, 24, 18, 14, 11, 7, 8, 10, 15, 18, 23]) )
entropy.calconditional()
print(entropy.entropyprob)
print(max(entropy.indentropies))


#print(entropy.prain)
#print(entropy.pshine)
#print(entropy.p_rain_g_month)
