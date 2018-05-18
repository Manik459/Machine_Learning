# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:07:00 2017

@author: MANIK
"""

from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from math import sqrt
import numpy as np
#import the dataset
d = pd.read_csv('winered.csv',sep=';')
X = d.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
y = d.iloc[:,11].values
print(d.head())
#split the dataset into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state=40)

#feature scaling
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
X_train_scaled = preprocessing.scale(X_train)
print(X_train_scaled)                

#Decision tree classifier
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

#predicting the test result set
y_pred = clf.predict(X_test)

#cross validation for 10 predictions

y_pred=clf.predict(X_test)
print(y_pred[0:10])
print(y_test[0:10])
        
 
#metrics 1
confidence = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)

# root mean squared error metric 
print("Mean squared Error : %.5f" % sqrt(mean_squared_error(y_test,y_pred)))

# mean absolute error metric for 
print("Mean absolute Error : %.5f" % mean_absolute_error(y_test,y_pred))



