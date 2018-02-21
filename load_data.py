# -*- coding: utf-8 -*-
import pandas as pd


df = pd.read_table("CleanedGames.csv", sep=";")

# Delete 'name' column
df = df.drop('Name', axis=1) # axis 1 is column

# Summary statistics
df.describe()

# N = number of games, M = number of attributes
N,M = df.shape

one_of_K_columns = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']
# get_dummies does one_out_of_K encoding.
X_encoded = pd.get_dummies(df, prefix=one_of_K_columns, columns=one_of_K_columns)
X = X_encoded.values
