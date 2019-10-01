# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:07:05 2019

@author: PC10
"""
from scipy.sparse import csr_matrix
import numpy as np
from sparse_dot_topn import awesome_cossim_topn as dot_product
from sklearn.preprocessing import normalize

def cosine_sparse(csr1,csr2, topn=10, min_value = 0.4, dense = True, similarity = True,expected_density = 1):
    """Computes the cosine distance between the rows of `csr`,
    smaller than the cut-off distance `epsilon`.
    """   
    if not dense:
        csr1 = csr_matrix(csr1).astype(bool,copy=False).astype(int,copy=False)
        csr1 = csr1.astype('float64',copy=False)
        csr1 = normalize(csr1, norm='l2', axis=1)
        csr2 = csr_matrix(csr2).astype(bool,copy=False).astype(int,copy=False)
        csr2 = csr2.astype('float64',copy=False)
        csr2 = normalize(csr2, norm='l2', axis=1)
        intrsct = dot_product(csr1,csr2.T, topn, min_value,expected_density)
        intrsct.data[intrsct.data>=1] = 1
        if not similarity:
                intrsct.data = 1-intrsct.data
    else:
        csr1 = csr_matrix(csr1).astype('float64',copy=False)
        csr1 = normalize(csr1, norm='l2', axis=1)
        csr2 = csr_matrix(csr2).astype('float64',copy=False)
        csr2 = normalize(csr2, norm='l2', axis=1)
        intrsct = dot_product(csr1,csr2.T, topn, min_value,expected_density)       
        if not similarity:
            intrsct.data = 1-intrsct.data
    return intrsct
