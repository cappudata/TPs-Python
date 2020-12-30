#1

list1 = []
for i in range(100):
    list1.append(i)
print(list1)

#2

list2 = []
for j in range(100,200):
    list2.append(j)
print(list2)

#3

for elt in list2:
    list1.append(elt)
print(list1)

#4

list1.reverse()
print(list1)

#5
# afficher list1 dans l'ordre croissant avec modifier la list

#list1.sort()
#print(list1)

#afficher list1 dans l'ordre croissant sans modifier la list

print(sorted(list1))

#6

list3 = []

for k in range(101):
    if (k % 2 == 0):
        list3.append(k)
print(list3)

#7

for elt in list3:
    if (elt % 10 ==0):
        list3.remove(elt)
print(list3)

#8

print(len(list3))
#print(list3.__sizeof__())    #la taille en memoire

#9

#for el in list3:
 #   if (el % 8 ==0):
  #      print(el)

#10
print("Question 10:")

for ind in range(20):
    print(list3[ind])

print("---")
print(list3[0:20])

#11
print("Question 11")

for inds in range(20,40):
    print(list3[inds])

print("---")
print(list3[20:41])

#12
print("Question 12")

for indi in range(-10,0):
    print(list3[indi])

print("---")
print(list3[-10:])

#13
print("Question 13")

list4 = [valeur for valeur in range(10) if (valeur % 2 == 1)]
print(list4)

#14

list5 = []
n1 = 1
n2 = 1
list5.append(n1)
list5.append(n2)

for indic in range(0,11):
    suivant = n1 + n2
    list5.append(suivant)
    n1 = n2
    n2 = suivant

print(list5)

#15
print("Question 15:")

def carre(x):
    x = x*x
    return x

def parite(x):
    if (x % 2 == 0):
        return True
    else: return False

def callback (function, list):
    list6 = []
    for elt in list:
        list6.append(function(elt))
    return  list6

print(callback(carre,list1))
print(callback(parite,list1))

#16

print("Question 16:")
def pair(x):
    if (x % 2 == 0):
        return True
    else: return False


def callback_2 (function, list):
    list7 = []
    for elt in list:
        if (function(elt)):
            list7.append(elt)
    return list7

print(callback_2(pair,list1))

#17
print("Exo 17:")

list_test = [1,2,3,4]
def somme(x1,x2):
    x1 = x1 + x2
    return x1

def produit (x1,x2):
    x1 = x1 * x2
    return x1

def callback_3 (function, list):
    list8 = []
    indi = 1
    for i in range(len(list)):
        for j in range(indi, len(list)):
           if (i != j):
                val = function(list[i], list[j])
                list8.append(val)
        indi += 1
    return list8

print(callback_3(somme,list_test))
print(callback_3(produit,list_test))

#18
print("Question 18:")

list_exo = [10,18,14,20,12,16]

def minMaxMoy(list):
    Min = list[0]
    Max = list[0]
    Somme = 0
    Moy = 0
    for i in range(len(list)):
        if (list[i] < Min):
            Min = list[i]
        if (list[i] > Max):
            Max = list[i]
        Somme += list[i]
    Moy = Somme/len(list)
    return Min, Max, Moy
    #print("Min:" + str(Min) + " / Max:" + str(Max) + " / Moyenne:" + str(Moy))


min,max,moy = minMaxMoy(list_exo)
print("Min:" + str(min))
print("Max:" + str(max))
print("Moyenne:" + str(moy))

ficher = open("minMaxMoy.txt","w")
ficher.write("Min:" + str(min))
ficher.write(" / Max:" + str(max))
ficher.write(" / Moyenne:" + str(moy))
ficher.close()

#ou d'autre moyen pour afficher.
#minMaxMoy(list_exo) --> juste enlever le commentaire #print dans la function et c bon

#19

print("Question 19:")

liste6 = [1,2,3]
liste6bis = liste6.copy()
#solution_2: liste6bis_2 = liste6[:]

liste6bis.append(4)

print(liste6)
print(liste6bis)
#solution 2: print(liste6bis_2)




jours= ["lundi","mardi","mercredi", "jeudi", "vendredi", "samedi", "dimanche"]



for i, jour in enumerate(jours):
    print(i,jour)


print(jours[1:3])
print(jours[2:])

personnes = { 12:"pierre", 58:"Paul"}

personnes[3] = "Jacques"

for k,v in personnes.items():
    print(k,v)


import  numpy as np

myliste = [[5,10,15],[25,25,30],[30,40,45]]
arr = np.array(myliste)
indice = np.array([[2,0]])

print(myliste)
print(np.argmax(myliste))
print(np.argmax(myliste,0))