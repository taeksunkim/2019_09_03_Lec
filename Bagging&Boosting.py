
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost


def loadDataset(filename):
    dataset=pd.read_csv(filename)
    input_features=['Age','Income','Limit','Cards','Student','Education']
    target_feature=['Balance']
    target_feature=['Balance']
    dataset=dataset[input_features+target_feature]
    dataset['Student'].replace('Yes',1,inplace=True)
    dataset['Student'].replace('No',0,inplace=True)
    x = dataset.drop('Balance', axis = 1).values
    y = dataset['Balance'].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
    return X_train, X_test, y_train, y_test

def getAccuracy(testSet, predictions):
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testSet, predictions)))
    print('test plot')
    plt.plot(testSet, predictions,  'ro', label='test set')
    plt.plot(testSet, testSet, label='standard line')
    plt.xlabel("Target")
    plt.ylabel("Output")
    plt.legend()
    plt.show()


# In[20]:


def RandomForest():
    X_train, X_test, y_train, y_test=loadDataset('Credit.csv')
    

    ###Fill out here
    
    getAccuracy(y_test, y_pred)
    
RandomForest()


# In[24]:


def xgb():
    X_train, X_test, y_train, y_test=loadDataset('Credit.csv')
    
    ###Fill out here

    
    getAccuracy(y_test, y_pred)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

xgb()

