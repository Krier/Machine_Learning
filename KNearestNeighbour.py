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

def Knearest():
    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        for j in range(0,len(y_est)):
            errors[j+i*len(y_est),l-1] = np.sum(y_est[j]!=y_test[j])
    


# Maximum number of neighbors
L=100

#CV = model_selection.LeaveOneOut()
KOuter = 2
KInner = 10

CVOuter = model_selection.KFold(n_splits=KOuter,shuffle=True)
CVInner = model_selection.KFold(n_splits=KInner,shuffle=True)


genErrors = np.zeros((KOuter,1))
minNumNeighbours = np.zeros((KOuter,1))
k=0
i=0
X = X.values

for train_index1, test_index1 in CVOuter.split(X, y):
    errors = np.zeros((len(train_index1),L))
    i=0
    XOuterTrain = X[train_index1,:]
    YOuterTrain = y[train_index1]
    XOuter_test = X[test_index1,:]
    yOuter_test = y[test_index1]
    
    for train_index2, test_index2 in CVInner.split(XOuterTrain, YOuterTrain):
        #print('Crossvalidation fold: {0}/{1}'.format(i+1,N)) 
        print('Crossvalidation fold: {0}/{1}'.format(i+1,KInner*KOuter))    
    
        # extract training and test set for current CV fold
        X_train = X[train_index2,:]
        y_train = y[train_index2]
        X_test = X[test_index2,:]
        y_test = y[test_index2]
        Knearest()
               
        i+=1 
        
    figure()
    plot(100*sum(errors,0)/(N/KOuter))
    xlabel('Number of neighbors')
    ylabel('Classification error rate (%)')
    show()
    minNumNeighbours[k] = np.argmin(100*sum(errors,0)/(N/KOuter))
    
    # test the model ( train on Dpar)
    knOuterclassifier = KNeighborsClassifier(n_neighbors=minNumNeighbours[k]);
    knOuterclassifier.fit(XOuterTrain, YOuterTrain);
    y_est = knclassifier.predict(XOuter_test);
    genErrors[k] = np.sum(y_est!=yOuter_test)
    k += 1
    
figure()
plot(minNumNeighbours)
xlabel('model number')
ylabel('optimal number of neighbours')
show()

figure()
plot(100*genErrors/(N/KOuter))
xlabel('Modelnumber')
ylabel('Error')
show()
#Plot the classification error rate
    

