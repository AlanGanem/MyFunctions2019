# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:58:54 2019

@author: PC10
"""

import numpy as  np
def func_slope(Y,axis):    
    
    X = np.arange(1,Y.shape[2]+1)
    X = X.reshape(1,1,X.shape[0],1)
    return  ((X*Y).mean(axis=axis) - X.mean()*Y.mean(axis=axis)) / ((X**2).mean() - (X.mean())**2)
