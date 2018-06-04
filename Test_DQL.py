# -*- coding: utf-8 -*-
"""
Created on Sun May 20 22:51:45 2018

@author: Paul
"""


from DoubleQLearning import DoubleQLearning
from KMaze_Bare import KMaze_Bare
import matplotlib.pyplot as plt
import numpy as np

N = 10 
K = 5
epsilon = 0.01
batch_size   = 8192 
switch_count = 500
DQL = DoubleQLearning(K,1,1,np.array([i for i in range(2*K)]),batch_size=batch_size,switchCount=switch_count)


EvalPeriod = 100000
EvalsPerPeriod = 100
EpochsPerTraining = 100
MaxSeqLength = 100

trainSequences = []
successSequence = []
for hjk in range(EvalPeriod):
    Npos = 0
    for CurrEval in range(EvalsPerPeriod):
        print('{}/{}                \r'.format(CurrEval+1,EvalsPerPeriod),end='')
        dim_chosen  = np.random.randint(0,2*K)
        env         = KMaze_Bare(N,K,dim_chosen)
        done = False
        observation = env.agentPos.copy()
        currentSequence = []
        while not done and len(currentSequence) <= MaxSeqLength:
            u = np.random.uniform()
            formerState = observation
            if u < epsilon:
                action = np.array([np.random.randint(0,2*K)])
            else:
                action,maxValue = DQL.getBestAction(observation.reshape(1,-1),np.array(dim_chosen).reshape(1,-1))

            observation, reward, done, info = env.step(action); observation=observation.copy()
            currentSequence.append((formerState,action,observation))
        if len(currentSequence) <= MaxSeqLength:
            if reward >= 0.5:
                Npos += 1
            processedSeq = env.seq2SeqNR(currentSequence)
            trainSequences += processedSeq
    successSequence.append(Npos * 1.0 / EvalsPerPeriod)
    print('Fitting Again.')
    DQL.fit(np.array(trainSequences),EpochsPerTraining)
    plt.figure()
    plt.plot(successSequence)
    plt.show()
    
    
    
