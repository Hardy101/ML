# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib as plt

# Importing the Dataset
df = pd.read_csv('Position_Salaries.csv')
X = df[:, 1:-1].values
y = df[:, :-1].values

# Training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_rg = LinearRegression(X, y)