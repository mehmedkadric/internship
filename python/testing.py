from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def ucitajDS():
    dataset = read_csv('noviDatasetTest.csv')
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
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Tacnost je: ", accuracy_score(y_test, y_pred))
    return


main()
