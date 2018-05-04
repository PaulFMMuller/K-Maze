# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:25:22 2018

@author: Paul
"""

from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Dense
from keras.models import Model,load_model
import numpy as np


class UVFA:
    # h_type : 'dot' [Future : 'MLP', ...)]
    def __init__(self,rank,n_layers,test_size=0.8,h_type='dot'):
        self.rank = rank
        self.n_layers = n_layers
        self.NMF = NMF(n_components=rank)
        if h_type == 'dot':    
            self.h = np.dot
              
    
    # We assume here that all gs are different.
    def fit(self,QValueMatrixes,g):
        mat = QValueMatrixes[0].reshape(-1,1)
        states = np.array(list(np.ndenumerate(QValueMatrixes)))
        states = np.asarray(list(states[:,0]))
        for i in range(1,len(QValueMatrixes)):
            mat = np.concatenate((mat,QValueMatrixes[i].reshape(-1,1)),axis=1)
        
        W = self.NMF.fit_transform(mat)
        H = np.transpose(self.NMF.components_)
        
        self.WNetwork = self.createNeuralNetwork(W.shape[1])
        self.HNetwork = self.createNeuralNetwork(H.shape[1])
        
        Wtrain,Wtest,Strain,Stest = train_test_split(W,states,test_size=0.2)
        Htrain,Htest,Gtrain,Gtest = train_test_split(H,g,test_size=0.2)
        
        # Use of an ensemble of NNs should be better. To be investigated.
        callBack = ModelCheckpoint('tempModel.mod', monitor='val_loss', save_best_only=True)
        self.WNetwork.fit(Strain,Wtrain,batch_size=32,validation_data=(Stest,Wtest),callbacks=[callBack])
        self.WNetwork = load_model('tempModel.mod')
        
        self.HNetwork.fit(Gtrain,Htrain,batch_size=32,validation_data=(Gtest,Htest),callbacks=[callBack])
        self.HNetwork = load_model('tempModel.mod')

        
    def createNeuralNetwork(self,inputShape):
        inputVar = Input(shape=(self.inputShape,))
        
        h = Dense(self.inputShape,activation='relu')(inputVar)
        for i in range(self.n_layers-2):
            h = h = Dense(self.inputShape,activation='relu')(h)
        res = Dense(self.rank,activation='linear')(h)
        
        modele = Model(inputs = inputVar,outputs=res)
        modele.compile(optimizer='Adam',loss='mean_squared_error',metric='mean_squared_error')
        return modele
        
        
    def predict(self,state,g):
        Phi = self.WNetwork.predict(state)
        Psi = self.HNetwork.predict(g)
        
        return (self.h(Phi,Psi))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    