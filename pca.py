# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import svd
from scipy.stats import zscore

from load_data import *

# Subtract mean value from data, divide by standard deviation to standardize
# Y = ( X - np.ones((N,1))*X.mean(0) )
Y = zscore(X, axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

#%%

# Z is the data projection onto the principal components
Z = Y @ V.T

plt.plot(Z[0], Z[1], '.')
