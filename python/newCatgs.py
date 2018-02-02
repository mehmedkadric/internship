from pandas import read_csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
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
    return np.unique(dataset["Age"])


def age(dataset):
    noveGodine = []
    prosjekGodina = np.mean(dataset["Age"])
    godine = dataset["Age"]
    maxGodine = max(godine)
    brojac = [0, 0, 0, 0, 0, 0]
    child = [0, 0]
    teen = [0, 0]
    adult = [0, 0]
    mature = [0, 0]
    old = [0, 0]
    for i, j in enumerate(godine):
        if str(j) == "nan":
            noveGodine.append(2)
            continue
        if j < 8:
            noveGodine.append(0)
            if dataset["Survived"][i] == 1:
                child[0] += 1
            else:
                child[1] += 1
            brojac[1] += 1
            continue
        if j < 20:
            noveGodine.append(1)
            if dataset["Survived"][i] == 1:
                teen[0] += 1
            else:
                teen[1] += 1
            brojac[2] += 1
            continue
        if j < 47:
            noveGodine.append(2)
            if dataset["Survived"][i] == 1:
                adult[0] += 1
            else:
                adult[1] += 1
            brojac[3] += 1
            continue
        if j < 60:
            noveGodine.append(3)
            if dataset["Survived"][i] == 1:
                mature[0] += 1
            else:
                mature[1] += 1
            brojac[4] += 1
            continue
        if j <= maxGodine:
            noveGodine.append(4)
            if dataset["Survived"][i] == 1:
                old[0] += 1
            else:
                old[1] += 1
            brojac[5] += 1
    """"
    print(child[0]/sum(child)*100)
    print(teen[0] / sum(teen) * 100)
    print(adult[0] / sum(adult) * 100)
    print(mature[0] / sum(mature) * 100)
    print(old[0] / sum(old) * 100)
    print(noveGodine)"""
    return noveGodine


def putnickaKlasa(dataset):
    pclass = []
    for i, j in enumerate(dataset["PClass"]):
        if j != '1st' and j != '2nd' and j != '3rd':
            pclass.append("2nd")
        else:
            pclass.append(dataset["PClass"][i])
    return pclass


def ishod(dataset):
    prezivjeli = []
    for i, j in enumerate(dataset["Survived"]):
        prezivjeli.append(j)
    return prezivjeli


def uklopi(a, b, c, d):
    new = []
    for i, j in enumerate(a):
        e = [a[i], b[i], c[i], d[i]]
        new.append(e)
    return new

def uklopiX(a, b, c):
    new = []
    for i, j in enumerate(a):
        e = [a[i], b[i], c[i]]
        new.append(e)
    return new


def uklopiY(a):
    new = []
    for i, j in enumerate(a):
        e = [a[i]]
        new.append(e)
    return new



dataset = ucitajDS()
titule = izdvojiTitule()
print(titule[0:10])
pclass = putnickaKlasa(dataset)
print(pclass[0:10])
noveGodine = age(dataset)
print(noveGodine[0:10])
prezivjeli = ishod(dataset)
print(prezivjeli[0:10])
newDataset = uklopi(titule, pclass, noveGodine, prezivjeli)
print(newDataset[0:2])

