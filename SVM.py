
# coding: utf-8

# In[3]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import random
from sklearn.svm import SVC
dataset=pd.read_csv('iris.csv')
data=dataset.loc[(dataset['variety']=='Virginica') | (dataset['variety']=='Versicolor')]

# data['variety'].replace('Virginica',1,inplace=True)
# data['variety'].replace('Versicolor',0,inplace=True)
# data['variety'].replace('Virginica',1,inplace=True)
# data['variety'].replace('Versicolor',0,inplace=True)


# In[5]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

x = data.drop('variety', axis = 1).values
y = data['variety'].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 3)

svm = SVC(kernel='linear', C=1.0, random_state=0)
# svm = SVC(kernel='rbf', C=1, random_state=0, gamma=10)  #gaussian kernel


svm.fit(X_train, y_train)
y_pred_svc = svm.predict(X_test)
print('Accuracy: %.2f' % getAccuracy(y_test, y_pred_svc))



