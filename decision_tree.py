# -*- coding: utf-8 -*-
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot

import numpy as np
from load_data_classification import X, y, N, M, C
from sklearn import tree, model_selection
import graphviz

attributeNames = X.columns.values
X = X.values

# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 30, 1)

def inner_cv(X_i, y_i):
    # K-fold crossvalidation
    K = 10
    CV = model_selection.KFold(n_splits=K,shuffle=True)
    # Initialize variable
    Error_train = np.empty((len(tc),K))
    Error_test = np.empty((len(tc),K))
    k=0
    for train_index, test_index in CV.split(X_i):
        print('Computing inner CV fold: {0}/{1}..'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train, y_train = X_i[train_index,:], y_i[train_index]
        X_test, y_test = X_i[test_index,:], y_i[test_index]
    
        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train,y_train.ravel())
            y_est_test = dtc.predict(X_test)
            y_est_train = dtc.predict(X_train)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(y_est_test != y_test)) / float(len(y_est_test))
            misclass_rate_train = sum(np.abs(y_est_train != y_train)) / float(len(y_est_train))
            Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
        k+=1
    
        
    f = figure()
    boxplot(Error_test.T)
    xlabel('Model complexity (max tree depth)')
    ylabel('Test error across CV folds, K={0})'.format(K))
    
    f = figure()
    plot(tc, Error_train.mean(1))
    plot(tc, Error_test.mean(1))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate, CV K={0})'.format(K))
    legend(['Error_train','Error_test'])
    show()
    return Error_test
    



K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)
# Initialize variable
Error_test = np.empty(K)
M_star = np.empty(K)

k=0
for train_index, test_index in CV.split(X):
    print('Computing outer CV fold: {0}/{1}..'.format(k+1,K))
    
    #Perform inner crossvalidation
    Validation_error = inner_cv(X[train_index], y[train_index])
    E_gen_s = Validation_error.mean(1) #TODO: Is this correct?
    # Select best model
    M_star[k] = tc[np.argmin(E_gen_s)]
    
    # extract training and test set for current CV fold
    X_test, y_test = X[test_index,:], y[test_index]
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=M_star[k])
    dtc = dtc.fit(X_test, y_test.ravel())
    y_est_test = dtc.predict(X_test)
    
    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = sum(np.abs(y_est_test != y_test)) / float(len(y_est_test))
    Error_test[k] = misclass_rate_test
    k+=1

Generalization_error = Error_test.mean()

# %%
#out = tree.export_graphviz(dtc, out_file='tree_deviance.gvz', feature_names=attributeNames)
#src=graphviz.Source.from_file('tree_deviance.gvz')
#src.render('../tree_deviance', view=True)