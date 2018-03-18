from pandas import read_csv
from pandas import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def ucitajDS():
    dataset = read_csv('noviDataset.csv')
    dataset.columns = ['Title', 'PClass', 'LifeStage', 'Survived']
    return dataset


def train(dataset, tuning=False):
    number = LabelEncoder()
    dataset["Title"] = number.fit_transform(dataset["Title"].astype('str'))
    dataset["PClass"] = number.fit_transform(dataset["PClass"].astype('str'))
    train, test = train_test_split(dataset, test_size=0.3)
    X_train = list(zip(train["Title"], train["PClass"], train["LifeStage"]))
    y_train = list((train["Survived"]))
    X_test = list(zip(test["Title"], test["PClass"], test["LifeStage"]))
    y_test = list((test["Survived"]))
    if tuning:
        penalty = ['l1', 'l2']
        C_range = [0.001,0.01,0.1,1,10,100]
        parameters = [{'C': C_range, 'penalty': penalty}]

        grid = GridSearchCV(LogisticRegression(), parameters, cv=5)
        grid.fit(X_train, y_train)

        bestC = grid.best_params_['C']
        bestP = grid.best_params_['penalty']
        print("The best parameters are: cost=", bestC, " and penalty=", bestP, "\n")

        print("Accuracy: {0:.3f}".format(accuracy_score(grid.predict(X_test), y_test)))
        print(grid.best_params_)
        print('\n')
        # Call best_estimators attribute
        print(grid.best_estimator_)
        y_pred_grid = grid.predict(X_test)
        print(classification_report(y_test, y_pred_grid))
        model_accuracy = round(accuracy_score(y_test, y_pred_grid) * 100, 2)
        print('Accuracy', model_accuracy, '%')
        return grid
    else:
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        print(classification_report(y_test,y_pred))
        logreg_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        print('Accuracy', logreg_accuracy, '%')
        return logreg


def prepareNewDataset():
    number = LabelEncoder()
    datasetTest = read_csv('noviDatasetTest.csv', header=None)
    datasetTest.columns = ['Title', 'PClass', 'LifeStage', 'Survived']
    datasetTest["Title"] = number.fit_transform(datasetTest["Title"].astype('str'))
    datasetTest["PClass"] = number.fit_transform(datasetTest["PClass"].astype('str'))
    return datasetTest


def main():
    dataset = ucitajDS()
    logreg = train(dataset, True)
    datasetTest = DataFrame(prepareNewDataset())
    X = list(zip(datasetTest["Title"], datasetTest["PClass"], datasetTest["LifeStage"]))
    Y = list((datasetTest["Survived"]))
    results = logreg.predict_proba(X)
    print(results)
    y_new = logreg.predict(X)
    print(Y, y_new)
    print(logreg.score(X,Y))

    return


main()

