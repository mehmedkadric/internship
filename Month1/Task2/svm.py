from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.svm import SVC
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
        param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        grid = GridSearchCV(SVC(probability=True),param_grid, refit = True, verbose=1)
        grid.fit(X_train,y_train)
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
        model = svm.SVC(kernel='linear', C=1, gamma=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        model_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        print('Accuracy', model_accuracy, '%')
        return model


def prepareNewDataset():
    number = LabelEncoder()
    datasetTest = read_csv('noviDatasetTest.csv', header=None)
    datasetTest.columns = ['Title', 'PClass', 'LifeStage', 'Survived']
    datasetTest["Title"] = number.fit_transform(datasetTest["Title"].astype('str'))
    datasetTest["PClass"] = number.fit_transform(datasetTest["PClass"].astype('str'))
    return datasetTest

def main():
    dataset = ucitajDS()
    model = train(dataset, True)
    datasetTest = prepareNewDataset()
    X = list(zip(datasetTest["Title"], datasetTest["PClass"], datasetTest["LifeStage"]))
    Y = list((datasetTest["Survived"]))
    results = model.predict_proba(X)
    print(results)
    print(model.predict(X),Y)
    print(model.score(X,Y))
    return


main()
