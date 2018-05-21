# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:25:22 2018

@author: Paul
"""

from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Dense,Concatenate,Add
from keras.models import Model,load_model
from keras.optimizers import RMSprop
import numpy as np


class UQFA:
    # h_type : 'dot' [Future : 'MLP', ...)]
    def __init__(self,stateShape,actionShape,goalShape,\
                 n_layers_state=3,hidden_size=3,n_layers_tot=1,learning_rate=0.001):
  
        self.neuralNetwork  =   self.createNeuralNetwork(stateShape,actionShape,goalShape,\
                                                         n_layers_state,hidden_size,n_layers_tot,learning_rate)
        
        
    # We assume here that all gs are different.
    def fit(self,states,actions,goals,targets,n_epochs=1):
        self.neuralNetwork.fit([states,actions,goals],targets,n_epochs=n_epochs)


    def createNeuralNetwork(self,stateShape,actionShape,goalShape,n_layers_state,hidden_size,n_layers_tot,learning_rate):
        inputState  = Input(shape=(stateShape,))
        inputAction = Input(shape=(actionShape,))
        inputGoal   = Input(shape=(goalShape,))
        
        inputStateAction = Concatenate()([inputState,inputAction])
        
        N = stateShape + actionShape
        count = 0
        lastState = inputStateAction
        
        hSA = Dense(N,activation='relu')(inputStateAction)        
        for i in range(n_layers_state-1):
            hSA = Dense(N,activation='relu')(hSA)
            count += 1
            if count == 2:
                count = 0
                hSA = Add()([lastState,hSA])
                lastState = hSA                 # ResNet
        oSA  = Dense(hidden_size,activation='relu')(hSA)
        
        hFin = Concatenate([oSA,inputGoal])
        for i in range(n_layers_tot):
            hFin = Dense(hidden_size+goalShape,activation='relu')(hFin)
        hFin = Dense(1,activation='linear')
        
        modele = Model(inputs=[inputState,inputAction,inputGoal],outputs=hFin)
        OPT = RMSprop(lr=learning_rate)
        modele.compile(optimizer=OPT,loss='mean_squared_error',metric='mean_squared_error')
        return modele
        
    
    def predict(self,state,action,goal):
        return self.neuralNetwork.predict([state,action,goal])
    
    
    def save(self,filepath):
        self.neuralNetwork.save(filepath)
    
    
    def load(self,filepath):
        self.neuralNetwork = load_model(filepath)
    
    def staticLoad(filepath):
        res = UQFA(0,0,0)
        res.load(filepath)
        return res
    
    
class zeroValue:
    def __init__():
        pass
    
    def predict(states,actions,goals):
        return np.zeros(states.shape[0])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    