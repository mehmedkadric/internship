import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn import preprocessing
import io
import request
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
from scipy import spatial
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import VarianceThreshold


def loadDataset():
    #url = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    #dataset = pd.read_csv(url, delimiter=" ", header=None)
    #dataset.to_csv('secom.csv', header=None)
    dataset = pd.read_csv('secom.csv', header=None)
    return dataset


def loadLabels():
    """url = "http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    labels = pd.read_csv(url, delimiter=" ", header=None)
    labels.to_csv('secom_labels.csv', header=None)"""
    labels = pd.read_csv('secom_labels.csv', header=None)
    return labels


def preprocess(ds):
    for i in ds:
        ds[i].fillna((ds[i].mean()), inplace=True)
        if len(pd.unique(ds[i])) == 1:
            del ds[i]
    del ds[0]
    x = ds.values
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    ds = pd.DataFrame(x_scaled)
    return ds.loc[:, ds.var() > 0.03]


def DBscan(ds, target):
    mins = [4]
    epses = [1.73]
    for i in mins:
        for j in epses:
            model = DBSCAN(eps=j, min_samples=i)
            model.fit_predict(ds.values)
            print("trying...")
            print(score(model, target)*100, pd.unique(model.labels_))
            print(model.labels_[180:210])
            print(target[180:210])
            if score(model, target) >= 0.9337:
                print(i, j)
    ctr = [0, 0]
    for i in model.labels_:
        if i == -1:
            ctr[0] += 1
        else:
            ctr[1] += 1
    print(ctr, pd.unique(model.labels_))
    return model


def locModel(ds, target):
    model = LocalOutlierFactor(n_neighbors=5)
    y_pred = model.fit_predict(ds.values)
    yModel = np.array(y_pred)
    yTarget = np.array(target)
    err = 0
    for i, j in enumerate(yModel):
        if j != yTarget[i]:
            err += 1
    print((len(yModel)-err)/len(yModel))


def score(model, target):
    yModel = np.array(model.labels_)
    yTarget = np.array(target)
    err = 0
    for i, j in enumerate(yModel):
        if j != yTarget[i]:
            err += 1
    return (len(yModel)-err)/len(yModel)


def main():
    ds = loadDataset()
    ds = preprocess(ds)
    print(ds.shape)
    labels = loadLabels()
    target = labels[:][1].values
    for i, j in enumerate(target):
        if j == -1:
            target[i] = 0
        else:
            target[i] = -1
    #model = locModel(ds, target)
    model = DBscan(ds, target)
    #score(model, target)
    #print(np.array(target))
    #print(np.array(model.labels_))
    return


main()