# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:27:14 2018

@author: Paul
"""

from UQFA import UQFA,zeroValue
import numpy as np


class doubleQLearning:
    
    def __init__(self,stateShape,actionShape,goalShape,possibleActions,\
                 n_layers_state=3,hidden_size=3,n_layers_tot=1,learning_rate=0.001,\
                 epsilon = 0.01,switchCount=500,gamma=0.99):
        
        self.activeNetwork = UQFA(stateShape,actionShape,goalShape,n_layers_state,hidden_size,n_layers_tot,learning_rate)
        self.targetNetwork = zeroValue()
        self.count         = 0    
        self.switchCount   = switchCount
        self.possibleActions = possibleActions
        self.gamma         = gamma
    
    
    def predict(self,state,action,goal):
        return self.activeNetwork.predict(state,action,goal)
    
    
    def generateTargets(self,sequences,gamma):
        states  = []
        actions = []
        goals   = []
        targets = []
        
        for i in range(len(sequences)):
            currSeq = sequences[i]
            for j in range(len(currSeq)):
                currState,currAction,currGoal,currReward,newState = currSeq[j]
                maxAc,newQ = self.getBestAction(newState,currGoal,False)
                states.append(currState); actions.append(currAction); goals.append(currGoal)
                targets.append(currReward+gamma*newQ)
        
        states  = np.array(states)
        actions = np.array(actions)
        goals   = np.array(goals)
        targets = np.array(targets)
        return states,actions,goals,targets
    
    
    def getBestAction(self,state,goal,realEval=True):
        maxValue  = -np.inf
        maxAction = -1
        if realEval:
            network = self.activeNetwork
        else:
            network = self.targetNetwork

        for action in self.possibleActions:
            tempVal = network.predict(state,action,goal)
            if tempVal > maxValue:
                maxValue  = tempVal
                maxAction = action
                
        return maxAction,maxValue
        
        
    def fit(self,sequences,epochs):
        states,actions,goals,targets = self.generateTargets(sequences,self.gamma)
        doneEpochs = min(epochs-self.count,self.switchCount-self.count)
        
        self.activeNetwork.fit(states,actions,goals,targets,doneEpochs)
        self.count += doneEpochs
        
        if self.count >= self.switchCount:
            self.count = 0
            self.activeNetwork.save('model.mod')
            self.targetNetwork = UQFA.staticLoad('model.mod')
            if epochs > doneEpochs:
                self.fit(sequences,epochs-doneEpochs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    