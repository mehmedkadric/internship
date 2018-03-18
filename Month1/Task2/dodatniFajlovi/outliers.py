from pandas import read_csv
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing


dataset = read_csv('Titanic_dataset.csv', usecols=['Name', 'PClass', 'Age', 'Sex', 'Survived'])
dataset.columns = ['Name', 'PClass', 'Age', 'Sex', 'Survived']
print("Unikatan broj putnickih klasa: ", np.unique(dataset["PClass"]))
brojac = 0
i = 0
for line in dataset["PClass"]:
    i = i + 1
    if str(line) == "*":
        brojac += 1
        print(i)

ages = []
for line in dataset["Age"]:
    if str(line) == "nan":
        continue
    else:
        ages.append(line)

#print("Broj putnika sa nepoznatom putničkom klasom: ", brojac)
#print("Broj ponavljanja klase *:", brojac)

print("Unikatan broj godina putnika: ", np.unique(ages))
print("Zscore za godine je između:", min(stats.zscore(ages)), " i ", max(stats.zscore(ages)))
print("Unikatan skup spolova: ", np.unique(dataset["Sex"]))
print("Unikatan skup ishoda: ", np.unique(dataset["Survived"]))
print(dataset["Name"].head(20))