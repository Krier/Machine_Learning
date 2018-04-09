# -*- coding: utf-8 -*-
#from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
import matplotlib.pyplot as plt

import numpy as np
from load_data_classification import X, y, N, M, C
from sklearn import tree, model_selection
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

#import graphviz

attributeNames = X.columns.values
X = X.values

#DecisionTree 
# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 30, 1)

#K nearest neighbors
# Maximum number of neighbors
L=40

KInner = 5
KOuter = 2

def Knearest(X_test, y_test, X_train, y_train, innerLoopNum):
    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        for j in range(0,len(y_est)):
            Error_test_KNearest[j+innerLoopNum*len(y_est),l-1] = np.sum(y_est[j]!=y_test[j])
            

def DecisionTree(X_test, y_test, X_train, y_train):
    Error_test = np.zeros(tc.size)
    Error_train = np.zeros(tc.size)
    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = sum(np.abs(y_est_test != y_test)) / float(len(y_est_test))
        misclass_rate_train = sum(np.abs(y_est_train != y_train)) / float(len(y_est_train))
        Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train
    
    return Error_test, Error_train

#def NaiveBayers(X_test, y_test, X_train, y_train):
#    Error_test = np.zeros(nb_alpha.size)
#    Error_train = np.zeros(nb_alpha.size)
#    for i, a in enumerate(nb_alpha):
#        nb_classifier = MultinomialNB(alpha=a, fit_prior=True)
#        nb_classifier.fit(X_train, y_train)
#        y_est_test_prob = nb_classifier.predict_proba(X_test)
#        y_est_train_prob = nb_classifier.predict_proba(X_train)
#        y_est_test = np.argmax(y_est_test_prob,1)
#        y_est_train = np.argmax(y_est_train_prob,1)
#        Error_test[i] = np.sum(y_est_test!=y_test)/y_test.size   
#        Error_train[i] = np.sum(y_est_train!=y_test)/y_test.size  
#    return Error_test, Error_train

def inner_cv(X_train, y_train, X_test, y_test):
    # K-fold crossvalidation
    K = 10
    CV = model_selection.KFold(n_splits=KInner,shuffle=True)
    # Initialize variable
    Error_train_tree = np.empty((len(tc),KInner))
    Error_test_tree = np.empty((len(tc),KInner))
#    Error_test_nb = np.zeros((nb_alpha.size, KInner))
#    Error_train_nb = np.zeros((nb_alpha.size, KInner))
    
    k=0
    for train_index, test_index in CV.split(X_train, y_train):
        print('Computing inner CV fold: {0}/{1}..'.format(k+1,KInner))
    
        # extract training and test set for current CV fold
        X_inner_train, y_inner_train = X[train_index,:], y[train_index]
        X_inner_test, y_inner_test = X[test_index,:], y[test_index]
    
        Error_test_tree[:,k], Error_train_tree[:,k] = DecisionTree(X_inner_test, y_inner_test, X_inner_train, y_inner_train)
#        Error_test_nb[:,k], Error_train_nb[:,k] = NaiveBayers(X_inner_test, y_inner_test, X_inner_train, y_inner_train)
        Knearest(X_inner_test, y_inner_test, X_inner_train, y_inner_train, k)
        
        k+=1
    
        
    f = plt.figure()
    plt.boxplot(Error_test_tree.T)
    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('Test error across CV folds, K={0})'.format(KInner))
    
    f = plt.figure()
    plt.plot(tc, Error_train_tree.mean(1))
    plt.plot(tc, Error_test_tree.mean(1))
    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('Error (misclassification rate, CV K={0})'.format(KInner))
    plt.legend(['Error_train','Error_test'])
    plt.show()
    return Error_test_tree
    
#K = 5
CV = model_selection.KFold(n_splits=KOuter,shuffle=True)
# Initialize variable
minNumNeighbours = np.empty(KOuter)
M_star = np.empty(KOuter)

Error_Test_Tree = np.empty(KOuter)
Error_Test_KNearest = np.zeros((KOuter,1))

Ttest_K_Nearest = np.empty((KOuter,1))
Ttest_DTree = np.empty((KOuter,1))

k=0
for train_index, test_index in CV.split(X, y):
    print('Computing outer CV fold: {0}/{1}..'.format(k+1,KOuter))
    
    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    Error_test_KNearest = np.zeros((len(train_index),L))
    
    #Perform inner crossvalidation
    Validation_error = inner_cv(X_train, y_train, X_test, y_test)
    E_gen_s = Validation_error.mean(1) #TODO: Is this correct?
    # Select best model
    M_star[k] = tc[np.argmin(E_gen_s)]

    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=M_star[k])
    dtc = dtc.fit(X_train, y_train.ravel())
    y_est_test = dtc.predict(X_test)
    
    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = sum(np.abs(y_est_test != y_test)) / float(len(y_est_test))
    Error_Test_Tree[k] = misclass_rate_test
    
    Ttest_DTree[k] = 100*(y_est_test!=y_test).sum().astype(float)/len(y_test)
    
    # knearest neighbor
    ###############################################
    plt.figure()
    plt.plot(100*sum(Error_test_KNearest,0)/(N/KOuter))
    plt.xlabel('Number of neighbors')
    plt.ylabel('Classification error rate (%)')
    plt.show()
    minNumNeighbours[k] = np.argmin(100*sum(Error_test_KNearest,0)/(N/KOuter))
#    
    # test the model ( train on Dpar)
    n = int(minNumNeighbours[k])
    knclassifier = KNeighborsClassifier(n_neighbors=n);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    Error_Test_KNearest[k] = np.sum(y_est!=y_test)/float(len(y_est))
    Ttest_K_Nearest[k] = 100*(y_est!=y_test).sum().astype(float)/len(y_test)
    ###############################################
    
    k+=1

Gen_error_Tree = Error_Test_Tree.mean()
Gen_error_KNearest = Error_Test_KNearest.mean()

# knearest neighbor plot
################################################
plt.figure()
plt.plot(minNumNeighbours)
plt.xlabel('model number')
plt.ylabel('optimal number of neighbours')
plt.show()

plt.figure()
plt.locator_params(nticks=KOuter)
plt.plot(100*Error_Test_Tree)
plt.plot(100*Error_Test_KNearest)
plt.xlabel('Modelnumber')
plt.ylabel('Classification error rate (%)')
plt.legend(['Decision Tree','K Kearest Neighbours'])
plt.show()
################################################

[tstatistic, pvalue] = stats.ttest_ind(Ttest_K_Nearest,Ttest_DTree)

if pvalue > 0.05 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
  

# %%
#out = tree.export_graphviz(dtc, out_file='tree_deviance.gvz', feature_names=attributeNames)
#src=graphviz.Source.from_file('tree_deviance.gvz')
#src.render('../tree_deviance', view=True)