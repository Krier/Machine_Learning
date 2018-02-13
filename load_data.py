# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df = pd.read_table("CleanedGames.csv", sep=";")

# Delete 'name' column
df = df.drop('Name', axis=1) # axis 1 is column

# Summary statistics
df.describe()

# N = number of games, M = number of attributes
N,M = df.shape