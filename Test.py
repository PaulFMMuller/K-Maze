# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:44:06 2018

@author: Paul
"""

from KMaze_Bare import KMaze_Bare
from UVFA import UVFA
import numpy as np

N   = 3  # Number of points
K   = 5   # Number of dimensions
gamma = 0.99

matrixList = []
goals = []
for goal in range(2*K):
    env = KMaze_Bare(N,K,goal)
    matrixList.append(env.getVMatrix(gamma))
    goals.append(goal)
goals = np.array(goals).reshape(-1,1)

n_layers = 3
n_epochs = 500
maxRank  = 2*K
errorByRank = np.zeros(maxRank)
for rank in range(1,maxRank):
    Qchap = UVFA(rank,n_layers,n_epochs)
    Qchap.fit(matrixList,goals)
    errorByRank[rank] = Qchap.getAveragePredictionError(matrixList,goals) # Checking whether the matrix overfits
    
    
# We are having some cool results !   
# Let's see what gives with Extrapolation !
    
    
n_layers = 3
n_epochs = 100
rank = 4
Ntries = 10

# For each goal, we will train the UVFA on every other possible goal
# and then check the results of the UVFA on that new goal.
# We'll make 10 tests, since the UVFA's results are 
# dependent on the NN's stochastic initialization.

errors = np.zeros((2*K,Ntries))
for goalK in range(2*K):
    matrixList = []
    goals = []
    for goal in range(2*K):
        if not goal == goalK:
            env = KMaze_Bare(N,K,goal)
            matrixList.append(env.getVMatrix(gamma))
            goals.append(goal)
            
    goals = np.array(goals).reshape(-1,1)
    env = KMaze_Bare(N,K,goalK)
    testMatrix = [env.getVMatrix(gamma)]
    testGoal   = np.array([goalK]).reshape(-1,1)
    for trial in range(Ntries):
        Qchap = UVFA(rank,n_layers,n_epochs)

        Qchap.fit(matrixList,goals)
        errors[goalK,trial] = Qchap.getAveragePredictionError(testMatrix,testGoal) # Checking whether the matrix overfits































