import pandas as pd
import numpy as np

#1
print("Question 1:")

arr = np.ones((3,3))
col = ['A','B','C']
ligne = ["i1", "i2", "i3"]

df = pd.DataFrame(data=arr, columns=col, index=ligne)
print(df)

#2
print("Question 2:")

arr_2 = np.ones((100,4))
col_2 = ["p1", "p2", "p3", "p4"]
ligne_2 = []

for x in range(1,101):
    string = str("i") + str(x)
    ligne_2.append(string)
print(ligne_2)

df_2 = pd.DataFrame(data=arr_2, columns=col_2, index=ligne_2)
print(df_2)

#3
print("Question 3:")


#4
print("Question 4:")

print(df_2.shape)

#5
print("Question 5:")

print(df_2.head(10))

#6
print("Question 6:")

print(df_2.tail(10))

#7
print("Question 7:")

print(df_2.columns.values)

#8
print("Question 8:")

print(df_2.columns)
print("----")
print(df_2.dtypes)

#9
print("Question 9:")

print(df_2.info())

#10
print("Question 10:")

print(df.describe())

#11float64
print("Question 11:")

print(df_2.p4)
print("<----->")
print(df_2["p4"])

#12
print("Question 12")

print(df_2[["p1","p4"]])

#13
print("Question 13")


#14
print("Question 14:")

print(df_2.apply(lambda x:x.mean(), axis=0))
print(df_2.apply(lambda x:x.mean(), axis=1))

# o- col
# 1 - ligne