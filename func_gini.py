# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:45:19 2019

@author: ganem
"""
import numpy as np
from matplotlib import pyplot as plt

def gini(array):    
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    #array = array.flatten()
    # Values must be sorted:
    array = np.sort(array)
    if array.size ==0:
        return 0
        
    
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001   
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def lorenz(arr):
    arr = np.sort(arr)
    # this divides the prefix sum by the total sum
    # this ensures all the values are between 0 and 1.0
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    # this prepends the 0 value (because 0% of all people have 0% of all wealth)
    return np.insert(scaled_prefix_sum, 0, 0)

def lorenz_plot(arr):    
    plt.clf()
    lorenz1 = lorenz(arr)
    plt.plot(np.linspace(0.0, 1.0, lorenz1.size), lorenz1)
    #plot the straight line perfect equality curve
    plt.plot([0,1], [0,1])
    plt.show()
    return

def ganem_plot(arr):
    lorenz1 = lorenz(arr)
    x = np.linspace(0.0, 1.0, lorenz1.size) 
    y = lorenz1
    x0 = np.argmin(((1-np.gradient(lorenz1, 1/(lorenz1.size-1)))**2))/(lorenz1.size-1)
    y0 = lorenz1[np.argmin(((1-np.gradient(lorenz1, 1/(lorenz1.size-1)))**2))]
    gini_ = gini(arr)
    k = y0/x0
    x0c = (k*x0+y0+gini_)/(1+k)
    y0c = (k*x0+y0+gini_)/(1+k) - gini_
    ganem_index = ((1-x0c)/(1-gini_))
    plt.plot([0,x0c], [0,y0c], 'ro-')
    plt.plot([x0c,1], [y0c,1], 'ro-')
    plt.plot([0,x0], [0,y0], 'ro-',color = 'y')
    plt.plot([x0,1], [y0,1], 'ro-',color = 'y')
    return

def ganem(arr):
    from func_gini import lorenz
    from func_gini import gini
    arr.sort()
    lorenz1 = lorenz(arr)
    x = np.linspace(0.0, 1.0, lorenz1.size) 
    y = lorenz1
    x0 = np.argmin(((1-np.gradient(lorenz1, 1/(lorenz1.size-1)))**2))/(lorenz1.size-1)
    y0 = lorenz1[np.argmin(((1-np.gradient(lorenz1, 1/(lorenz1.size-1)))**2))]
    gini_ = gini(arr)
    k = y0/(x0+0.000001)
    x0c = (k*x0+y0+gini_)/(1+k)
    y0c = (k*x0+y0+gini_)/(1+k) - gini_
    ganem_index = ((1-x0c)/(1-gini_))    
    return [ganem_index,x0c,y0]

