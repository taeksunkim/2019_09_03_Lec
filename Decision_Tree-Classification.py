
# coding: utf-8

# # Practical 3 : Tree Based Method

# ## 1.Tree Implementation

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import operator


# In[2]:


def loadDataset(filename):
    data=pd.read_csv(filename)
    trainingSet, testSet = train_test_split(data, test_size = 0.25, random_state = 3)
    return trainingSet.values, testSet.values

training_set,test_set=loadDataset('iris.csv')


# In[10]:


verbose = False
def getVotes(data):
    classVotes = {} #dict
    for i in range(len(data)):
        response = data[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    return classVotes

def Entropy(splits):
    entropy = 0
    for split in splits:
        if(len(split) != 0):
            vote=getVotes(split)
            total=sum(vote.values())
    #### Fill out here, We need to calculate entropy for each split.
    
    return entropy


# In[11]:


def split(index, value, data):
    left_split = [element for element in data if(element[index]<value)] #TODO condition
    right_split = [element for element in data if(element[index]>=value)] #TODO condition
    return [left_split, right_split]


# In[17]:


def split_tester(data): #find optimal split
    optimal_split_ind, optimal_split_value, optimal_residual, optimal_splits = -1,-1,float("inf"),[]
    for curr_ind in range(data.shape[1]-1): #for all features
        min_val=np.min(data[:,curr_ind])
        for curr_val in data: #for all values in the data
            if curr_val[curr_ind] == min_val:
                continue
            if(verbose):print("Curr_split : " + str((curr_ind, curr_val[curr_ind])))
            split_result = split(curr_ind, curr_val[curr_ind], data) #TODO (comments : get the current split)
            
            if(verbose):print(split_result)
            residual_value = Entropy(split_result)#TODO (comments : get the RSS of the current split)
            
            if(verbose):print("Residual : " + str(residual_value))
            if residual_value < optimal_residual:
                optimal_split_ind, optimal_split_value, optimal_residual, optimal_splits = curr_ind,                                                                    curr_val[curr_ind], residual_value, split_result
                
    return optimal_split_ind, optimal_split_value, optimal_splits   # index is feature, value is for crietria, splits is data list.


# In[18]:


def tree_building(data, min_size): #minimun data size in a split
    if(data.shape[0] > min_size): #building tree until the minimum.
        ind, value, [left, right] = split_tester(data) #using optimal criteria using split_tester
        left, right = np.array(left), np.array(right)
        return [tree_building(left, min_size), tree_building(right, min_size),ind,value]
    else:
        return data  #output is the data in a leaf node.


# In[19]:


def getResponse(data):
    classVotes = {} #dict
    for i in range(len(data)):
        response = data[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# In[20]:


def predict(tree, input_vector):#recursive until reaching the leaf node.
    if(type(tree[-1]) != np.float): #when reach the leaf node. tree represent the split data.
        if(len(tree) == 1):  #when number of data is 1
            return(tree[0][-1])  
        else:
            return ###Fill out here using getResponse function.
    else:  #before reaching leaf node
        left_tree, right_tree, split_ind, split_value = tree #information of the current split 
        if(input_vector[split_ind]<split_value): #which split the input data belong to
            return predict(left_tree, input_vector)
        else:
            return predict(right_tree, input_vector)
    


# In[21]:


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


tree = tree_building(training_set,10)
predictions=[]
for employee in test_set:
    predictions.append(predict(tree,employee))
#     print("Predicted : " + str(predict(tree,employee)) + ", Actual : " + str(employee[-1]))
accuracy = getAccuracy(test_set, predictions)
print('Accuracy: ' + repr(accuracy))

