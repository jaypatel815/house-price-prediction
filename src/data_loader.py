import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

# housing = fetch_california_housing(as_frame=True)
# df = housing.frame

# Basic statistics about the dataset
# print(df.describe())

# Check for missing values
# print(df.isnull().sum())

# Visualize relationships between features and the target variable
# plt.figure(figsize=(10, 6))
# plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3)
# plt.xlabel('Median Income')
# plt.ylabel('Median House Value')
# plt.title('House Value vs Income')
# plt.show()

# Histogram of house prices
# df['MedHouseVal'].hist(bins=50, figsize=(10, 6))
# plt.xlabel('Median House Value')
# plt.ylabel('Frequency')
# plt.title('Distribution of House Prices')
# plt.show()
