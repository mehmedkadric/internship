from pandas import read_csv
import pandas as pd
import numpy as np


dataset = read_csv('Titanic_dataset.csv', usecols=['Name', 'PClass', 'Age', 'Sex', 'Survived'])
dataset.columns = ['Name', 'PClass', 'Age', 'Sex', 'Survived']

print("Broj putnika: ", len(dataset))
brojac = 0
for line in dataset["Name"]:

    if str(line)=="nan":
       brojac += 1

print("Broj putnika sa nepoznatim imenom: ", brojac)

brojac = 0
for line in dataset["PClass"]:

    if str(line)=="nan":
       brojac += 1

print("Broj putnika sa nepoznatom putničkom klasom: ", brojac)

brojac = 0
for line in dataset["Age"]:

    if str(line)=="nan":
       brojac += 1

print("Broj putnika sa nepoznatim godinama: ", brojac)



brojac = 0
for line in dataset["Sex"]:

    if str(line)=="nan":
       brojac += 1

print("Broj putnika sa nepoznatim spolom: ", brojac)

brojac = 0
for line in dataset["Survived"]:

    if str(line)=="nan":
       brojac += 1

print("Broj putnika sa nepoznatim ishodom (prezivjeli ili ne): ", brojac)
