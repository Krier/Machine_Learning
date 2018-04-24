# -*- coding: utf-8 -*-
import pandas as pd
import math
import numpy as np


df = pd.read_table("CleanedGames.csv", sep=";")
df_orig = df
# Delete 'name' column
df = df.drop('Name', axis=1) # axis 1 is column
df = df.drop('Developer', axis=1) # axis 1 is column
df = df.drop('Global_Sales', axis=1) # axis 1 is column

#summary statistics
df.describe()

# N = number of games, M = number of attributes
N,M = df.shape

one_of_K_columns = ['Platform', 'Rating', 'Publisher']
# get_dummies does one_out_of_K encoding.
X_encoded = pd.get_dummies(df, prefix=one_of_K_columns, columns=one_of_K_columns)

# divide one_out_of_K encoded with the sqrt 
for j in one_of_K_columns :
    indicies = [item for item in X_encoded.columns if item.startswith(j)]
    for k in indicies:
        X_encoded[k] = X_encoded[k].map(lambda a : a/math.sqrt(len(indicies)))        

X = X_encoded
X = X.drop('Genre', axis=1)
classLabels = df.Genre
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

y = np.array([classDict[value] for value in classLabels])
N,M = X.shape
C = len(classNames)        
# [i for i in X_encoded.values[:,1589] if i > 0]

#df_orig.Publisher.value_counts().index.tolist()
#[i if i not in singles else 'Other' for i in df.Publisher]