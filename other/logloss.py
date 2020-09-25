#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:03:15 2019

@author: saul
"""
#This is an example of how to calculate logloss for Machine Learning
#!/usr/bin/env python
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.datasets import make_moons
from math import log
import matplotlib.pyplot as plt
import numpy as np

probs = []
logprobs = []
def logloss(true_label, predicted, eps=1e-15):
    p =  np.clip(predicted, eps, 1 - eps)
    print("tt")
    print("P values is :", p)
    probs.append(p)
    #print (true_label)
    if true_label == 1:
        print("True Label is 1")
        logprobs.append(-log(p))
        return -log(p)
        
    else:
        logprobs.append(-log(1 - p))
        return -log(1- p)
    
if __name__ == "__main__":
    print("tttttt")
    
    for i in np.arange(100, 0, -0.1):
        
        print(logloss(1, i))
        
plt.plot(probs, logprobs)
plt.xlabel('probabilities')
plt.ylabel('log probabilities')
plt.show()
