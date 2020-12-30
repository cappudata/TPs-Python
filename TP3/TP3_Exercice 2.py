import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#1
print("Question 1:")

data = pd.read_csv("diabetes.csv", sep=",")
print(data)

#2
print("Question 2:")

#print(data.head(10))

#3
print("Question 3:")

#4
print("Question 4:")

def euclidian_distance(v1,v2):
    return math.sqrt(((v1-v2)**2).sum())

#5
print("Question 5:")

diabete_1 = data.head(1)
print(diabete_1)

diabete_2 = data.iloc[0,:]
print(diabete_2)

diabete_3 = data.iloc[10,:]
print(diabete_3)

print(euclidian_distance(diabete_2,diabete_3))
#print(euclidian_distance(diabete_1,diabete_3)) -- ca function pas car c pas la meme representation

#6
print("Question 6:")

def neighbors (dataframe, diabete_test, val):

    new_list = []
    for i in range(dataframe.shape[0]):
        distance = euclidian_distance(diabete_test, dataframe.iloc[i,:2])
        new_list.append(distance)

    dataframe["Distance"] = new_list
    data = dataframe.sort_values(by="Distance")

    return data.head(val)

#7
print("Question 7:")

def prediction (dataframe,diabete_test,k):
    data= neighbors(dataframe, diabete_test,k)
    data_2 = data["Outcome"].value_counts()
    return data_2.idxmax()

#8
print("Question 8:")

echant_60 = data.sample(frac=0.6, random_state=1)
print(echant_60)

echant_40 = data[~data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']].isin(echant_60).any(axis=1)]
print(echant_60)

#9
print("Question 9:")

def taux(dataframe, data_test,k):
    taux = 0
    for i in range(data_test.shape[0]):
        pred = prediction(dataframe, data_test.iloc[i],k)
        if ( pred == data_test.iloc[i]['Outcome']):
            taux = taux - 1
    resul = (taux / data_test.shape[0])*100
    return  resul

#10
print("Question 10:")
#ici jai compris qu'il faut compter le nombre de personne avec Outcome = 1

compter = (data["Outcome"]).value_counts()
positif_2 = compter[1]
print(compter)
print(positif_2)

#11
print("Question 11:")
#ici compter le nombre de personne avec Outcome =0

negatif = compter[0]
print(negatif)

#12
print("Question 12:")

#difference valeur de K.
#je ne sais pas pourquoi ca m'affiche pas
print(taux(echant_60,echant_40,5))
print(taux(echant_60,echant_40,20))
print(taux(echant_60,echant_40,40))
print(taux(echant_60,echant_40,80))