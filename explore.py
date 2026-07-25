"""This piece of code helps to explore the dataset"""

import pandas as pd  # library for loading/working with tabular data

# Load the training and testing datasets into tables (DataFrames)
train = pd.read_csv("data/training.csv")
test = pd.read_csv("data/testing.csv")

print(train.head())              # preview first 5 rows
print(train.shape, test.shape)   # (rows, columns) for both files
print(train['class'].value_counts())  # counts class labels
