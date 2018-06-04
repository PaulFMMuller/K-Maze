# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:27:14 2018

@author: Paul
"""

from UQFA import UQFA,zeroValue
import numpy as np


class DoubleQLearning:
    
    def __init__(self,stateShape,actionShape,goalShape,possibleActions,\
                 n_layers_state=3,hidden_size=3,n_layers_tot=1,learning_rate=0.001,\
                 epsilon = 0.01,switchCount=500,gamma=0.99,batch_size=32):
        
        self.activeNetwork = UQFA(stateShape,actionShape,goalShape,n_layers_state,hidden_size,n_layers_tot,learning_rate)
        self.targetNetwork = zeroValue()
        self.count         = 0    
        self.switchCount   = switchCount
        self.possibleActions = possibleActions
        self.gamma         = gamma
        self.batch_size    = batch_size 
    
    
    def predict(self,state,action,goal):
        return self.activeNetwork.predict(state,action,goal)
    
    
    def generateTargets(self,sequences,gamma):
        states   = np.array(list(sequences[:,0]))
        action   = np.array(list(sequences[:,1]))
        goal     = np.array(list(sequences[:,2]))
        reward   = np.array(list(sequences[:,3]))
        newState = np.array(list(sequences[:,4]))

        maxAc,newQ = self.getBestAction(newState,goal,False)
        targets = reward+gamma*newQ
        
        return states,action,goal,targets
    
    
    def getBestAction(self,state,goal,realEval=True):
        if realEval:
            network = self.activeNetwork
        else:
            network = self.targetNetwork

        maxValue  = -np.inf * np.ones(state.shape[0])
        maxAction = -1 * np.ones(state.shape[0]).astype(int)
    
        for action in self.possibleActions:
            tempVals = network.predict(state,np.array([action]*len(maxValue)),goal).reshape(-1,)
            booleans = tempVals > maxValue
            
            maxValue[booleans] = tempVals[booleans]
            maxAction[booleans] = action
                
        return maxAction,maxValue
        
        
    def fit(self,sequences,epochs):
        states,actions,goals,targets = self.generateTargets(sequences,self.gamma)
        doneEpochs = min(epochs,self.switchCount-self.count)
        
        self.activeNetwork.fit(states,actions.reshape(-1,1),goals.reshape(-1,1),targets.reshape(-1,1),doneEpochs,self.batch_size)
        self.count += doneEpochs
        
        if self.count >= self.switchCount:
            self.count = 0
            self.activeNetwork.save('model.mod')
            self.targetNetwork = UQFA.staticLoad('model.mod')
            if epochs > doneEpochs:
                self.fit(sequences,epochs-doneEpochs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    