from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

def ucitajDS():
    dataset = read_csv('noviDataset.csv')
    dataset.columns = ['Title', 'PClass', 'LifeStage', 'Survived']
    return dataset

def main():
    dataset = ucitajDS()
    number = LabelEncoder()
    dataset["Title"] = number.fit_transform(dataset["Title"].astype('str'))
    dataset["PClass"] = number.fit_transform(dataset["PClass"].astype('str'))
    train, test = train_test_split(dataset, test_size=0.3)
    X_train = list(zip(train["Title"], train["PClass"], train["LifeStage"]))
    y_train = list((train["Survived"]))
    X_test = list(zip(test["Title"], test["PClass"], test["LifeStage"]))
    y_test = list((test["Survived"]))
    model = svm.SVC(kernel='linear', C=1, gamma=1)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    datasetTest = read_csv('noviDatasetTest.csv', header=None)
    datasetTest.columns = ['Title', 'PClass', 'LifeStage', 'Survived']
    datasetTest["Title"] = number.fit_transform(datasetTest["Title"].astype('str'))
    datasetTest["PClass"] = number.fit_transform(datasetTest["PClass"].astype('str'))
    X = list(zip(datasetTest["Title"], datasetTest["PClass"], datasetTest["LifeStage"]))
    Y = list((datasetTest["Survived"]))
    print(model.score(X, Y))
    return


main()
