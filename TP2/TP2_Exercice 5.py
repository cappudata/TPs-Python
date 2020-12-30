import pandas as pd
import matplotlib.pyplot as plt

#1
print("Question 1:")
redwine = pd.read_csv("redwinePD.csv", sep=";")
whitewine = pd.read_csv("whitewinePD.csv", sep=";")

print(redwine)

#2
print("Question 2:")

print(redwine.sort_values(by="alcohol").tail(5))
print(whitewine.sort_values(by="alcohol").tail(5))

#3
print("Question 3:")

print(redwine.loc[redwine.quality > 7])
print(whitewine.loc[whitewine["quality"] > 7])

#4
print("Question 4:")


#5
print("Question 5:")

print(redwine.loc[(redwine["quality"] > 7) & (redwine["alcohol"] < 10)])
print(whitewine.loc[(whitewine["quality"] >7) & (whitewine["alcohol"] <9)])

#6
print("Question 6:")

pro = redwine["quality"]
#propo = pro.value_counts().plot.pie(autopct = lambda x: str(round(x, 2)) + '%')
#plt.show()

#propo = pro.value_counts()
#print(propo)

#7
print("Question 7:")

alcol = redwine["alcohol"]
quali = redwine["quality"]

#plt.scatter(alcol,quali)
#plt.show()

#8
print("Question 8:")

acidity = redwine["volatile acidity"]
print(acidity)

#9
print("Question 9:")

redwine.boxplot(column="alcohol",by="quality")
plt.show()
whitewine.boxplot(column="alcohol",by="quality")
plt.show()

#10
print("Question 10:")

#alcol_3 = redwine["alcohol"].value_counts()
#print(alcol_3)
#alcol_3.plot(kind='bar')
#plt.show()

#11
print("Question 11:")

fig, (alcol_3, alcol_4 ) = plt.subplots(1,2, figsize=(20,20))

alcol_3 = redwine["alcohol"].value_counts()
alcol_4 = redwine["quality"].value_counts()
print(alcol_4)
alcol_3.plot(kind='bar')
alcol_4.plot(kind='bar')
plt.show()

#12
print("Question 12:")

