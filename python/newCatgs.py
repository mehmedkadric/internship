from pandas import read_csv
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing
from nameparser import HumanName


def ucitajDS():
    dataset = read_csv('Titanic_dataset.csv')
    dataset.columns = ['Name', 'PClass', 'Age', 'Sex', 'Survived']
    return dataset


def izdvojiTitule():
    titles = []
    for row in dataset["Name"]:
        name = HumanName(row)
        titles.append(name.title)
    return titles


def prebrojTitule(titles, title):
    counter = 0
    for j in titles:
        if j == title:
            counter += 1
    return counter


def prebrojSve(titles, uniqueTitles):
    brojTitula = []
    for i, j in enumerate(uniqueTitles):
        brojTitula.append(prebrojTitule(titles, uniqueTitles[i]))
    #Vrati niz gdje su indeksi u relaciji sa nizom uniqueTitles
    return brojTitula


def ispisiStatsTitula(uniqueTitles, brojTitula):
    for i, j in enumerate(uniqueTitles):
        print("Titula '", uniqueTitles[i], "' se pojavljiva ", brojTitula[i], "puta")

    if dataset["Name"].size != sum(brojTitula):
        print("Nema smisla (broj ljudi vs broj titula): ", dataset["Name"].size, sum(brojTitula))
    else:
        print("Ima smisla (broj ljudi vs broj titula): ", dataset["Name"].size, sum(brojTitula))


def unikatneGodine(dataset):
    uniqueAges = np.unique(dataset["Age"])
    brojGodina = []
    for i, j in enumerate(dataset["Age"]):
        if str(j) == "nan":
            continue
        else:
            brojGodina.append(j)
    godine = dataset["Age"]
    maxGodine = max(godine)
    brojac = [0, 0, 0, 0]
    child = [0, 0]
    teen = [0, 0]
    adult = [0, 0]
    old = [0, 0]
    for i, j in enumerate(godine):
        if str(j) == "nan":
            brojac[3] += 1
            continue
        if j < 13:
            if dataset["Survived"][i] == 1:
                child[0] += 1
            else:
                child[1] += 1
            brojac[0] += 1
            continue
        if j < 31:
            if dataset["Survived"][i] == 1:
                adult[0] += 1
            else:
                adult[1] += 1
            brojac[1] += 1
            continue
        if j <= maxGodine:
            if dataset["Survived"][i] == 1:
                old[0] += 1
            else:
                old[1] += 1
            brojac[2] += 1
    #srednjaVrijednost = np.mean(brojGodina)
    print(child[0]/sum(child)*100)
    print(adult[0]/sum(adult)*100)
    print(old[0]/sum(old)*100)
    print(brojac)
    return uniqueAges


dataset = ucitajDS()
titles = izdvojiTitule()
uniqueTitles = np.unique(titles)
brojTitula = prebrojSve(titles, uniqueTitles)
#ispisiStatsTitula(uniqueTitles, brojTitula)

unikatneGodine(dataset)
