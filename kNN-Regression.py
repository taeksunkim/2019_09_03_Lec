
# coding: utf-8

# In[197]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import random
from sklearn.model_selection import train_test_split
import operator


# In[161]:



def loadDataset(filename):
    df = pd.read_csv(filename, index_col=0)
    input_features=['Age','Income','Limit','Cards','Student','Education']
    target_feature=['Balance']
    df_input=df[input_features+target_feature]
    
    #########fill out here
    #we need to change categorical variable (Yes or No) to dummy variable (1 or 0).


    train=df_input.sample(frac=0.75,random_state=3) #split into train and test
    test=df_input.drop(train.index)
    trainingSet=train.values.tolist()
    testSet=test.values.tolist()
    return trainingSet, testSet

def euclideanDistance(instance1, instance2): 

    #######fill out here same with previous 

def getNeighbors(trainingSet, testInstance, k):

    ######fill out here same with previous 


def getResponse(neighbors):
    ######fill out here same with previous 
    ####here we need to caculate the average target value of k nearest neighbors.
    
    return mean

def getAccuracy(testSet, predictions):
    accuracy=np.sqrt(np.average((np.array(testSet)[:,-1]-np.array(predictions))**2))
    print('test plot')
    plt.plot(np.array(testSet)[:,-1], np.array(predictions),  'ro', label='test set')
    plt.plot(np.array(testSet)[:,-1], np.array(testSet)[:,-1], label='standard line')
    plt.xlabel("Target")
    plt.ylabel("Output")
    plt.legend()
    plt.show()
    return accuracy


# In[160]:


def main():
# prepare data
    trainingSet,testSet=loadDataset('Credit.csv')
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy(RMSE): ' + repr(accuracy) )
    
main()

