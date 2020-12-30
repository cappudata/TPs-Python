#Exercice 1
print("Question 1:")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv("iris.csv", sep=",")
print(data)

df = data.values
print(df)
#2
print("Question 2:")

#print(data.head(10))

#3
print("Qurstion 3:")

couleurs = ["blue", "green", "red"]
for i in range(3):
    data_sbc = data.loc[data["species"] == i]
    plt.scatter(data_sbc["petal_length"],data_sbc["petal_width"], color=couleurs[i])

#plt.xlabel("petal_length")
#plt.ylabel("petal_ width")
#plt.show()

#4
print("Question 4:")
print(data.shape)

def euclidian_distance(v1,v2):
    return math.sqrt(((v1-v2)**2).sum())

#5
print("Question 5:")

iris_1 = data.head(1)
print(iris_1)

#iris_1 = data.iloc[:1,:2]
#print(iris_1)

iris_2 = data.iloc[0,:2]
print(iris_2)

iris_3 = data.iloc[10,:2]
print(iris_3)

print(euclidian_distance(iris_2, iris_3))

#6
print("Question 6:")

def neighbors (dataframe, iris, val):

    new_list = []
    for i in range(dataframe.shape[0]):
        distance = euclidian_distance(iris, dataframe.iloc[i,:2])
        new_list.append(distance)

    dataframe["Distance"] = new_list
    data = dataframe.sort_values(by="Distance")

    return data.head(val)

#7
print("Question 7:")

k = 3
x_test1 = np.array([3,0.6])
x_test2 = np.array([5,1.6])
x_test3 = np.array([1,0.6])
print(neighbors(data,x_test1,k))
print(neighbors(data,x_test2,k))
print(neighbors(data,x_test3,k))

#8
print("Question 8:")

def prediction (dataframe,iris,k):
    data= neighbors(dataframe, iris,k)
    data_2 = data["species"].value_counts()
    return data_2.idxmax()

print(prediction(data,x_test2,5))

#9
print("Question 9:")

echant = data.sample(frac=0.6, random_state=1)
x_echant = echant[["petal_length","petal_width"]]
y_echant = echant.species

#echant.to_csv("test1", sep=";")
#print(echant)


#reste = data[~data[['petal_length','petal_width','species','Distance']].isin(echant).any(axis=1)]
reste = data.drop(echant.index)

x_reste = reste[["petal_length","petal_width"]]
y_reste = reste.species

#reste.to_csv("test2", sep=";")
#print(reste)
#print(reste.iloc[1])


#10
print("Question 10:")

def taux(dataframe, test,k):
    taux = 0
    for i in range(test.shape[0]):
        pred = prediction(dataframe, test.iloc[i],k)
        if ( pred == test.iloc[i]['species']):
            taux = taux + 1
    resul = (taux / test.shape[0])*100
    return  resul

print(taux(echant,reste,10))
#11
print("Question 11:")

#def taux_espece(Date, Etest,k):
 #   res = []
  #  for i in range(3):
   #     df1 = Date.loc[Date['species'] == i]
    #    df2 = Etest.loc[Etest['species'] == i]
     #   print(df1)
      #  print(df2)
       # res.append(taux(df1,df2,k))
  #  return res

#print(taux_espece(echant,reste,10))


def taux_espece(dataframe,test,k):
    reussite1=0
    reussite2=0
    reussite3=0
    for i in range(test.shape[0]):
        nearest_neighbors = neighbors(echant, test.iloc[i], k)
        if (prediction(dataframe,nearest_neighbors,k) == y_reste.iloc[i]):
            if (y_reste.iloc[i] == 0):
                reussite1+=1
            elif (y_reste.iloc[i] == 1):
                reussite2+=1
            else:
                reussite3+=1
        return reussite1/y_reste.value_counts()[0]*100, \
               reussite2/y_reste.value_counts()[1]*100, \
               reussite3/y_reste.value_counts()[2]*100

print(taux_espece())


