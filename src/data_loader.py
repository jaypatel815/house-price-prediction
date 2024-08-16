import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing


def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df
