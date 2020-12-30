import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#1
np.random.seed(0)

#2
print("Question 2:")

#table = np.random.randint(0,500,500)
table1 = np.random.random(500)

#3
print("Question 3:")

""""
o = 15
u = 100
new_table = []

def func(x):
    y = x*15+100
    return y

for x in range(len(table1)):
    resul = func(table1[x])
    new_table.append(resul)

print(new_table)

#4 5 6
print("Question 4,5,6:")

plt.hist(new_table, bins=50, color="green")
plt.xlabel("Probabolité de densité")
plt.ylabel("Quotient intellectuel")
plt.show()
"""

#8
print("Question 8:")


print("---------")
#partie 2
print("Partie 2")

redwine = np.genfromtxt("redwineNP.csv", delimiter=";",)
whitewine = np.genfromtxt("whitewineNP.csv", delimiter=";")

