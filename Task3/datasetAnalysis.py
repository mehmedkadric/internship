import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn import preprocessing
import io
import request
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


def loadDataset():
    #url = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    #dataset = pd.read_csv(url, delimiter=" ", header=None)
    #dataset.to_csv('secom.csv', header=None)
    dataset = pd.read_csv('secom.csv', header=None)
    return dataset


def loadLabels():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    labels = pd.read_csv(url, delimiter=" ", header=None)
    labels.to_csv('secom_labels.csv', header=None)
    #labels = pd.read_csv('secom_labels.csv', header=None)
    return labels


def preprocess(ds):
    for i in ds:
        ds[i].fillna((ds[i].mean()), inplace=True)
        if len(pd.unique(ds[i])) == 1:
            del ds[i]
    x = ds.values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    ds = pd.DataFrame(x_scaled)
    return ds


def DBscan(ds):
    model = DBSCAN(eps=3, min_samples=20).fit(ds.values)
    br = [0, 0]
    for i in model.labels_:
        if i == 0:
            br[0] += 1
        else:
            br[1] +=1
    print(br, pd.unique(model.labels_))
    print(model.labels_[50:70])
    return model


def main():
    ds = loadDataset()
    ds = preprocess(ds)
    model = DBscan(ds)
    labels = loadLabels()
    target = labels[:][0].values
    for i, j in enumerate(target):
        if j == -1:
            target[i] = 0
        else:
            target[i] = -1

    print(target[50:70])
    return


main()