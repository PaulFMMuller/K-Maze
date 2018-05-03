# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:25:22 2018

@author: Paul
"""

from sklearn.decomposition import NMF
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
        H = self.NMF.components_
        
        
    def predict(self,state,g):
        pass