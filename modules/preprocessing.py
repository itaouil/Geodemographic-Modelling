"""
    Data preprocessing script.
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv("../dataset/Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical data (country and gender)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
x[:, 1] = labelEncoder.fit_transform(x[:, 1])
x[:, 2] = labelEncoder.fit_transform(x[:, 2])

# One hot encode the categorical data
oneHotEncoder = OneHotEncoder(categorical_features = [1])
x = oneHotEncoder.fit_transform(x).toarray()
x = x[:, 1:]

# Split dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test  = sc.transform(x_test.reshape(-1, 1))
