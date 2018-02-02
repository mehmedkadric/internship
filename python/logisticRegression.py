from pandas import read_csv
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from nameparser import HumanName
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


def ucitajDS():
    dataset = read_csv('Titanic_dataset.csv')
    dataset.columns = ['Name', 'PClass', 'Age', 'Sex', 'Survived']
    return dataset

dataset = ucitajDS()
dataset = dataset.dropna()
print(dataset.shape)
print(list(dataset.columns))
dataset.head()