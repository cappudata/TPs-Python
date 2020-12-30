print("Exercice 1")
import numpy as np


#6
print("Question 6:")

np.random.seed(0)

#2
print("Question 2:")

arr_2 = np.random.randint(0,100,100)
print(arr_2)

#3
print("Question 3:")

arr_3 = np.random.randn(100)
print(arr_3)

#4
print("Question 4:")
arr_4 = np.random.randn(25)
print(arr_4)

#5
print("Question 5:")

matric = np.random.randint(0,100,(10,10))
print(matric)

#Les entrées/Sorties
print("Les entrées/ Sorties")
fname = np.savetxt("matrice1.csv",matric,delimiter=";")


