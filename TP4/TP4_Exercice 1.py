import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#1
print("Question 1:")

data = np.loadtxt("digits.csv")
print(data)
print(data.shape)

X = data[:,:64]
Y = data[:,-1:]  #c un vecteur
print(Y)
print(Y.shape)

#2
print("Question 2:")

chiffre = X[0].reshape((8,8))
#plt.imshow(chiffre)
#plt.show()

#3
print("Question 3:")

#les valeurs d correspond au nombre des colonnes => 64  X.shape[1]
#les valeurs k correspond au nombre des classes. Ya 10 classes
#nombre de parametre d+1*k

K = np.unique(Y)
D = X.shape[1]
print("valeur de K: " + str(K.shape[0]))
print("valeur de D: " + str(D))

#4
print("Question 4:")

ecart_type = 0.01
moyen = 0

weights = np.random.randn(D+1, K.shape[0])*ecart_type + moyen
print(weights.shape)

#5
print('Question 5:')

matrice = np.ones((X.shape[0],1))
new_X = np.hstack((matrice,X)) #la matrice X avec la colonne de '1' ajoutée
print(new_X.shape)   #pour verifier si la colonne est bien ajouter dans la matrice X.

#6
print("Question 6:")

def output(matrice_entree_X, matrice_modele):
    V = np.dot(matrice_entree_X,matrice_modele)
    V_2 = np.exp(V)
    S = np.sum(V_2,1)
    F = V_2/S.reshape(S.shape[0],1)         #c la sortie
    return F

#7
print("Question 7:")

sortie = output(new_X,weights)
print(sortie)
print(sortie.shape)

print(np.sum(sortie, axis=1)) #pour vérifier

#8
print("Question 8:")


Y_one_hot = np.zeros((1797,10))

for i in range(Y_one_hot.shape[0]):
    Y_one_hot[i,int(Y[i])] = 1


print(Y_one_hot)
print(Y_one_hot.shape)

#9
print("Question 9:")

def crossentropy(matrice_prediction_F, matrice_y_one_hot):
    return - (matrice_y_one_hot*np.log(matrice_prediction_F) ).sum()/matrice_y_one_hot.shape[0]


def crossentropy_2(pref_f, y_one_hot):
    return - (y_one_hot*np.log(pref_f) + (1-y_one_hot*np.log(1-pref_f))).mean()
#10
print("QUestion 10:")

print(crossentropy(sortie,Y_one_hot))
print(crossentropy_2(sortie,Y_one_hot))

#11
print("Question 11:")


#12
print("Question 12:")

def prediction(matrice_f):
    return np.argmax(matrice_f,1)  #axix="1" pour les lignes

#13
print("Question 13:")

print(sortie.shape)
print(prediction(sortie))
print(prediction(sortie).shape)

#14
print("Question 14:")

def error_rate(label_predict, vrai_label):
    err = 0
    for i in range(label_predict.shape[0]):
        if(label_predict[i] != vrai_label[i]):
            err += 1
    return (err*100)/label_predict.shape[0]


#15
print("Question 15:")

print(error_rate((prediction(sortie)),Y))

#16
print("Question 16:")

def gradient(X,f,y_one_hot):
    gradient = (np.transpose(X) @ (f - y_one_hot))/X.shape[0]
    return gradient

#17
print("Question 17:")

indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]

X_train = X[training_idx]
Y_train = Y[training_idx]

X_test = X[test_idx]
Y_test = Y[test_idx]

#18
print("Question 18:")

print(Y_train)
print(Y_train.shape)

y_one_hot_train = np.zeros((1257,10))
for i in range(y_one_hot_train.shape[0]):
   y_one_hot_train[i,int(Y_train[i])] = 1

print(y_one_hot_train)

#19
print("Question 19:")

