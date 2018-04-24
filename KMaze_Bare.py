#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:18:35 2018

@author: paul
"""

import numpy as np

class KMaze_Bare:
    
    def __init__(self,N,K,chosenDim,initPos=None,random_seed=0,minReward=1/1000,maxReward=1):
        if (not initPos is None) and (not (len(initPos) == K)):
            raise Exception('Error : initial position array and dimensional number mismatch : K = {} ; len(initPos) = {}'.format(\
                            K,len(initPos)))
        if N <= 0:
            raise Exception('Error : Empty environment space.')
                
        np.random.seed(random_seed)
        if initPos is None:
            self.agentPos = np.random.randint(low=1,high=N-1,size=K)
        else:
            self.agentPos = np.array(initPos)
            
        self.N = N
        self.K = K
        self.chosenDim = chosenDim
        self.minReward = minReward
        self.maxReward = maxReward
        self.startingPosition = self.agentPos
        
        
    def render(self):
        if   self.K == 1:
            outputString = '.' * self.N
            outputString = list(outputString)
            outputString[self.agentPos[0]] = '0'
            outputString = ''.join(outputString)
        elif self.K == 2:
            newStrings = []
            outputString = ''
            for i in range(self.N):
                newStrings.append('.' * self.N)
            tempString = list(newStrings[self.agentPos[0]])
            tempString[self.agentPos[1]] = '0'
            newStrings[self.agentPos[0]] = ''.join(tempString)
            for i in range(self.N):
                outputString += newStrings[i] + '\n'
        else:
            outputString = 'Sorry, no way to visualize dim >= 3 environments yet ! ;)'
        print(outputString)
        
        
    def sample(self):
        return np.random.randint(low=0,high=2*self.K)
    
    
    def reset(self):
        self.agentPos = self.startingPos
        return self.agentPos
    
    
    def step(self,action):
        if action >= 2*self.K:
            raise Exception('Error : New action out of action space.')
            
        actionDim = action // 2                 # Find to which dimension the action is applied
        actionDir = 2*(action-2*actionDim)-1    # And whether we increase or decrease our position in this dimension.
        
        self.agentPos[actionDim] += actionDir
        observation = self.agentPos
        
        reward = 0                              # 0 by default
        info   = [None]
        done   = False
        if np.any(self.agentPos <= 0) or np.any(self.agentPos >= self.N):
            done = True
            if self.agentPos[self.chosenDim] >= self.N:
                reward = self.maxReward
            else:
                reward = self.minReward
            
        return observation, reward, done, info