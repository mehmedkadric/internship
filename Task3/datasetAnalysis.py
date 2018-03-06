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
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
import collections
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RandomizedSearchCV
from time import time


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
    #return ds


def distanceToTheNearestNeighbor(values):
    nbrs = NearestNeighbors(n_neighbors=5).fit(values)
    distances, indices = nbrs.kneighbors(values)
    distances.sort
    plt.hist(distances[:, 1], bins=30)
    plt.title('Histogram - the nearest neighbor distances')
    plt.show()
    return distances, indices


def numberOfPointsWithinDistance(values, eps):
    nbrs = NearestNeighbors(n_neighbors=500).fit(values)
    distances, indices = nbrs.kneighbors(values)
    distances.sort()
    newArr = [0 for x in range(len(values))]
    for i in range(0, len(values)):
        for j in range(0, 200):
            if distances[i][j] < eps:
                newArr[i] += 1
    plt.hist(newArr, bins=30)
    plt.title('Histogram - minPts in Eps')
    plt.show()
    return



def DBscan(ds, target):
    model = DBSCAN(eps=1.22, min_samples=5)
    model.fit_predict(ds.values)
    print(model.labels_)
    print(classification_report(target, model.labels_))
    print(collections.Counter(model.labels_))
    #numberOfPointsWithinDistance(ds.values, 1.2)
    print(model.labels_[0:20])
    print(target[0:20])
    return model


def locModel(ds, target):
    model = LocalOutlierFactor(n_neighbors=20)
    y_pred = model.fit_predict(ds.values)
    print(y_pred[0:20])
    print(target[0:20])
    print(classification_report(target, y_pred))
    return model


def SVManomaly(ds, target, g=0.0022, n=0.0045):
    dsOld = ds
    targetOld = target
    df1 = pd.DataFrame(target)
    df1.columns = ['labels']
    ds = pd.concat([ds, df1], axis=1)
    ds = ds.sort_values(by=['labels'])
    ds = ds.drop('labels', 1)
    ds = ds[0:1463]
    target = sorted(target)
    target = target[0:1463]
    X_train, X_test, y_train, y_test = train_test_split(ds, target, train_size=0.8)

    #g=110, nu=0.0015
    gs = [120]
    nus = [0.0019]
    model = OneClassSVM(gamma=120, nu=0.0019, kernel='rbf')
    model.fit(ds.values)
    preds = model.predict(dsOld.values)
    targs = targetOld
    """print(preds[2:22])
    print(targs[2:22])
    print("accuracy: ", metrics.accuracy_score(targs, preds))
    print("precision: ", metrics.precision_score(targs, preds))
    print("recall: ", metrics.recall_score(targs, preds))
    print("f1: ", metrics.f1_score(targs, preds))
    print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))"""
    print(confusion_matrix(preds, targs))
    print(targs[0:20])
    print(preds[0:20])
    print(classification_report(targs, preds))
    return model


def calcDist(ds, target):
    df1 = pd.DataFrame(target)
    df1.columns = ['labels']
    ds = pd.concat([ds, df1], axis=1)
    ds = ds.sort_values(by=['labels'])
    ds = ds.drop('labels', 1)
    a = pairwise_distances(ds.values, metric='euclidean')
    np.fill_diagonal(a, 5)
    outliers = a.min(axis=1)
    np.fill_diagonal(a, 0)
    print(np.shape(ds.values))
    #print(a[:1463][:1463].mean(axis=1))
    plt.hist(a[:1463][0], bins=50)
    plt.title("Minimal distances between normal examples")
    plt.show()
    """plt.hist(outliers[0:1463], bins=70)
    plt.show()
    plt.hist(outliers[-104:], bins=70)
    plt.show()
    #print(pwMatrix.min(axis=1))
    newDst = ds.values
    print(newDst)
    print(dsOld[i].values)
    print(newDst)"""

    #plt.hist(newDst)
    #plt.show()



def main():
    ds = loadDataset()
    ds = preprocess(ds)
    #print(ds.shape)
    labels = loadLabels()
    target = labels[:][1].values
    for i, j in enumerate(target):
        if j == -1:
            target[i] = 1
            continue
        else:
            target[i] = -1
    #model = locModel(ds, target)
    model = SVManomaly(ds, target)

    for i, j in enumerate(target):
        if j == -1:
            target[i] = 0
        else:
            target[i] = 1

    #calcDist(ds, target)

    for i, j in enumerate(target):
        if j == 0:
            target[i] = -1
        else:
            target[i] = 1

    #model = DBscan(ds, target)
    #score(model, target)
    #print(np.array(target))
    #print(np.array(model.labels_))
    return


main()