#Corrigé TD4 - Algo plus proches voisins

import numpy as np
import math
import pandas as pd

#1)
def euclidian_distance(v1,v2):

    distance = 0

    for i in range(v1.shape[0]):
        distance += (v1[i] - v2[i])**2

    return math.sqrt(distance)



#2)
dataset = [[2.7810836 ,2.550537003 ,0],
         [1.465489372 ,2.362125076 ,0],
         [3.396561688 ,4.400293529 ,0],
         [1.38807019 ,1.850220317 ,0],
         [3.06407232 ,3.005305973 ,0],
         [7.627531214 ,2.759262235 ,1],
         [5.332441248 ,2.088626775 ,1],
         [6.922596716 ,1.77106367 ,1],
         [8.675418651 ,-0.242068655 ,1],
         [7.673756466 ,3.508563011 ,1]]

#3)
data = np.array((dataset))

X_train = data[:,:2]
y_train = data[:,2]

# Distance entre les points 0 et 1
print(euclidian_distance(X_train[0,:],X_train[1,:]))

#4)
x_test = np.array([5,-0.2])

def neighbors(X_train, y_label, x_test, k):

    list_distances =  []

    for i in range(X_train.shape[0]):

        distance = euclidian_distance(X_train[i,:], x_test)

        list_distances.append(distance)


    df = pd.DataFrame()

    df["label"] = y_label
    df["distance"] = list_distances

    df = df.sort_values(by="distance")

    return df.iloc[:k,:]

#5)
k = 3
nearest_neighbors = neighbors(X_train, y_train, x_test, k)
print(nearest_neighbors)

#6)
def prediction(neighbors):

    mean = neighbors["label"].mean()

    if (mean > 0.5):
        return mean, 1
    else:
        return mean, 0


score, pred = prediction(nearest_neighbors)

print("pred " + str(pred) + ", score " + str(score))


