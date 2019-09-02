
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


# In[199]:


def loadDataset(filename):
    data=pd.read_csv(filename)
    trainingSet, testSet = train_test_split(data, test_size = 0.25, random_state = 3)
    return trainingSet.values.tolist(), testSet.values.tolist()


# In[224]:


def euclideanDistance(instance1, instance2): 
    distance=0
#####fill out here.
#instance has a form like [4.6, 3.2, 1.4, 0.2, 'Setosa']
#we need to calculate euclidean distance between instance1 and 2.
#the return value of distance is a scalar value.
    return distance


# In[ ]:


## test for eucldieanDistance function
instance1=[4.6, 3.2, 1.4, 0.2, 'Setosa']
instance2=[6.4, 2.7, 5.3, 1.9, 'Virginica']
value=euclideanDistance(instance1, instance2)
print('Function value: ',value)
print('Correct answer: ',4.646504062195578)








# In[225]:


def getNeighbors(trainingSet, testInstance, k): 
    neighbors=[]
#fill out here
#using the function of euclideanDistance we can get distances between training data and testInstance.
#sorting them with the distances and getting the k nearest neighbors.
#the return value of neighbors has a form like [[5.4, 3.4, 1.7, 0.2, 'Setosa'], [5.2, 3.4, 1.4, 0.2, 'Setosa'], [5.5, 3.5, 1.3, 0.2, 'Setosa']] when k=3
    return neighbors


# In[ ]:


## test for getNeighbors function

trainingSet, testSet=loadDataset('iris.csv')
testInstance=[4.6, 3.2, 1.4, 0.2, 'Setosa']
k=3
value=getNeighbors(trainingSet,testInstance,k)
print('Function value: ',value)
print('Correct answer: ',[[4.7, 3.2, 1.3, 0.2, 'Setosa'], [4.7, 3.2, 1.6, 0.2, 'Setosa'], [4.8, 3.0, 1.4, 0.3, 'Setosa']])





# In[226]:


def getResponse(neighbors):
    classVotes = {}  
    for i in range(len(neighbors)):
        response = neighbors[i][-1]  ##species of i-th neighbors.
        ###fill out here
        ### we need to make a dictionary, key is iris species and value is counting number of each species.
        

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    ##print(sortedVotes) has a form like [('Virginica', 2), ('Versicolor', 1)] when k=3.
    return sortedVotes[0][0] ##return value is string value of one species.
 


# In[ ]:


## test for getResponse function
neighbors=[[6.9, 3.1, 4.9, 1.5, 'Versicolor'], [6.5, 3.0, 5.2, 2.0, 'Virginica'], [6.5, 3.2, 5.1, 2.0, 'Virginica']]
value=getResponse(neighbors)
print('Function value:',value)
print('Correct answer: Virginica')

 
    
    


# In[227]:


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


# In[228]:


def main():
    trainingSet, testSet=loadDataset('iris.csv')
    print ('Train set: ',len(trainingSet))
    print ('Test set: ',len(testSet))
    # generate predictions
    predictions=[]
    k=3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    
main()

