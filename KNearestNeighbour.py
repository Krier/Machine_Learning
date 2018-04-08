#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:38:58 2018

@author: Jens
"""
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

from load_data_classification import *

# Maximum number of neighbors
L=40

#CV = model_selection.LeaveOneOut()
KOuter = 5
KInner = 10

CVOuter = model_selection.KFold(n_splits=KOuter,shuffle=True)
CVInner = model_selection.KFold(n_splits=KInner,shuffle=True)

errors = np.zeros((N,L))
minInnerError = np.zeros((KOuter,1))
k=0
i=0
X = X.values

for train_index1, test_index1 in CVOuter.split(X, y):
    errors = np.zeros((N,L))
    XOuterTrain = X[train_index1,:]
    YOuterTrain = y[train_index1]
    
    for train_index2, test_index2 in CVInner.split(XOuterTrain, YOuterTrain):
        #print('Crossvalidation fold: {0}/{1}'.format(i+1,N)) 
        print('Crossvalidation fold: {0}/{1}'.format(i+1,KInner*KOuter))    
    
        # extract training and test set for current CV fold
        X_train = X[train_index2,:]
        y_train = y[train_index2]
        X_test = X[test_index2,:]
        y_test = y[test_index2]
    
        # Fit classifier and classify the test points (consider 1 to 40 neighbors)
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train, y_train);
            y_est = knclassifier.predict(X_test);
            errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
            i+=1 
    figure()
    plot(100*sum(errors,0)/N)
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()
    minInnerError[k] = np.argmin(100*sum(errors,0)/N)
    k += 1
    # test the model
#    knclassifier = KNeighborsClassifier(n_neighbors=l);
#    knclassifier.fit(X_train, y_train);
#    y_est = knclassifier.predict(X_test);
#    errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    
figure()
plot(minInnerError)
xlabel('model number')
ylabel('optimal number of neighbours')
show()
#Plot the classification error rate
    

