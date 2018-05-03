# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:24:06 2018

@author: Paul
"""

from KMaze_Bare import KMaze_Bare

N   = 10  # 10 points
K   = 2   # Two dimensions
Dim = 2   # 

env = KMaze_Bare(N,K,Dim)

done      = False
userInput = True
PossibleActions = list(range(2*K))

env.render()
while not done:
    if userInput:
        action = int(input('Action : '))
        while action not in PossibleActions:
            action = int(input('Action not understood. It needs to be within {}.\nAction : '.format(PossibleActions)))
    else:
        action = env.sample()
        
    observation, reward, done, info = env.step(action)
    print(reward)
    env.render()
