import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn import preprocessing
import io
import request


def loadDataset():
    #url = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    #dataset = pd.read_csv(url, delimiter=" ", header=None)
    #dataset.to_csv('secom.csv', header=None)
    dataset = pd.read_csv('secom.csv', header=None)
    return dataset


def preprocessing(ds):
    for i in ds:
        ds[i].fillna((ds[i].mean()), inplace=True)
        if len(pd.unique(ds[i])) == 1:
            del ds[i]
    return ds


def main():
    ds = loadDataset()
    #print(ds.describe())
    ds = preprocessing(ds)

    return


main()