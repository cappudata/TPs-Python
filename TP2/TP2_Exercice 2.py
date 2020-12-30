import matplotlib.pyplot as plt
import numpy as np

#1
print("Question 1:")

t1 = [1,4,8,9]
t2 = [12,25,34,78]

#plt.scatter(t1,t2)
#plt.show()

#2
print("Question 2:")

arr = np.random.randint(0,1000,1000)

#plt.scatter(arr,range(0,1000))
#plt.show()

#3
print("Question 3:")

#plt.plot(t1,t2)
#plt.show()

#4
print("Question 4")

table = np.random.randn(100)

#plt.plot(table)
#plt.show()

#5 6 7 8 9 10
print("Question 5,6,7,8,9,10:")

x = np.linspace(0,2,100)

"""""
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,7))

ax1.plot(x, "blue",  label="Linéaire")
ax1.legend()


ax2.plot(x*x, "orange", label="Carée")
ax2.legend()

ax3.plot(x*x*x, "green", label="Cubique")
ax3.legend()

fig.suptitle("Comparaison des fonctions linéraire, quadratique et cubique")
plt.xlabel("abscisse")
plt.ylabel("ordonnée")
#plt.show()
"""""

#11
print("Question 11:")

t3 = np.random.uniform(0,10,100)
#from random import sample
#date = sample(range(1,1000),100)
plt.hist(t3)
plt.show()

