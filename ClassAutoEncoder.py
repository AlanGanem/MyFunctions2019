# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:33:11 2019

@author: PC10
"""

import keras
import sklearn
class ClassAutoEncoder():

    def __init__(self,X, compress_factor=None, latent_dim=None):
        minmaxscaler = sklearn.preprocessing.MinMaxScaler()
        self.X = minmaxscaler.fit_transform(X)
        self.compress_factor = compress_factor
        self.latent_dim = latent_dim
        self.dim = X.shape[1]
        
        self.X_train,self.X_val = sklearn.model_selection.train_test_split(self.X,random_state = 123, train_size = 0.8)
        try:
            assert sum([not isinstance(compress_factor,type(None)), not isinstance(latent_dim,type(None))]) == 1
            
        except AssertionError as e:
            e.args+=('one of compress_factor or latent_dim must be passed',)
            raise
        
        if not isinstance(compress_factor,type(None)):
            self.latent_dim =  int(self.compress_factor*self.dim)
        
        
        self.encoder_input = keras.layers.Input(shape = (self.dim,))
        first_layer = keras.layers.Dense(self.dim)
        compress_layer = keras.layers.Dense(self.latent_dim)
        output_layer = keras.layers.Dense(self.dim)
        
        
        self.first_encoder_layer = first_layer(self.encoder_input)
        self.encoder_output = compress_layer(self.first_encoder_layer)
        self.decoder_output = output_layer(self.encoder_output)
        
        self.full_model = keras.models.Model(self.encoder_input,self.decoder_output)
        self.encoder = keras.models.Model(self.encoder_input,self.encoder_output)
        