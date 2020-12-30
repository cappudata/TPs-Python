#les codes de cours
list_points = []

for i in range(5):
    for j in range(5):
        list_points.append((i,j))
print(list_points)


personnes = {12:"Pierre", 58:"Paul"}

print(personnes[12])

personnes[3] = "Jac"
print(personnes)

for k,v in personnes.items():
    print(k,v)

print("--------------")
#1
print("Question 1:")
import numpy as np

#2
print("Question 2:")

list = np.empty(10, dtype=str)
print("taille: " + str(len(list)))
print(list)

list_new2 = 10*['']
print(list_new2)

#3
print("Question 3:")

arr_2 = np.zeros(10)
print("taille: " + str(len(arr_2)))
print(arr_2)

#4
print("Question 4:")

arr_3 = np.ones(10)
print("taille: " + str(len(arr_3)))
print(arr_3)

#5
print("Question 5:")

arr_4 = np.ones(10)*5
print("1/ " + str(arr_4))

arr_42 = np.zeros(10)+5
print("2/ " + str(arr_42))

#6
print("Question 6:")

list_2 = ["1","2,","3", "4", "5","6","7","8","9","10"]
arr_5 = np.array(list_2)

print(arr_5)

#7
print("Question 7:")

tab1 = np.arange(1,10.5,0.5)
print(tab1)

#8
print("Question 8:")

tab2 = np.linspace(1,10,20)
print(tab2)

#9
print("Question 9:")

tab3 = np.random.randint(10,30,20)
print(tab3)

#10
print("Question 10:")

somme = np.sum(tab3)
print(somme)

#11
print("Question 11:")


#12
print("Question 12:")

moy = np.mean(tab3)
print(moy)

#13
print('Question 13:')

pro_sca = np.dot(tab2,tab3)
print(pro_sca)

#14
print("Question 14:")

tab3_test = tab3 > 20
print(tab3[tab3_test])

#15
print("Question 15:")

print(tab3[1])
print(tab3[5])
print(tab3[10])
print(tab3[15])

#Les Matrices
print("Les Matrices:")

#1
print("Question 1:")

matric = [[1,2,3],[4,5,6],[7,8,9]]
mat = np.array(matric)
print(mat)

#2
print("Question 2:")

matric_2 = np.ones((10,10))
print(matric_2)

#3
print("Question 3:")

print("Taille: " + str(matric_2.size))

#4
print("Question 4:")

print("Dimension: " + str(matric_2.ndim))

#5
print("Question 5:")

print("Forme: " + str(matric_2.shape))
print("Structure: " + str(type(matric_2)))

#6
print("Question 6:")


list_mat = np.arange(9)
print(list_mat)
print(list_mat.reshape((3,3)))

#7
print("Question 7:")

matric_A = (np.arange(1,101)/100).reshape(10,10)
print(matric_A)

#8
print("Question 8:")

matric_B = matric_A[6:,:4]
print(matric_B)

#9
print("Question 9:")

matric_B2 = np.ones((matric_B.shape))
print(matric_B2)

#10
print("Question 10:")

s1 = matric_A[2:4,:1]
s2 = matric_A[2:4,2:3]
s3 = matric_A[2:4,4:5]
s4 = matric_A[2:4,6:7]

A = np.concatenate((s1,s2), axis=1)
B = np.concatenate((s3,s4), axis=1)
matric_C = np.concatenate((A,B), axis=1)

print(matric_C)

#11
print("Question 11:")

produi_matric = np.dot(matric_C, matric_B)
print(produi_matric)

#12
print("Question 12: ")

for i in range(len(matric_A)):
    for j in range(len(matric_A)):
        if (matric_A[i][j] == 0.55):
            print(i,j)
            print(matric_A[i][j])

#13
print("Question 13:")

col = matric_A[:,4:5]
print(col)

#14
print("Question 14:")

ligne = matric_A[4:5,:]
print(ligne)

#16
print("Question 16:")

moy_matric = np.mean(matric_A)
print(moy_matric)

#17
print("Question 17:")

moy_ligne = np.mean(matric_A,axis=1)
print(moy_ligne)

#18
print("Qustion 18:")

moy_col = np.mean(matric_A,axis=0)
print(moy_col)

#19
print("Question 19:")

Col_1 = matric_A[:,:1]
#print(Col_1)

Ligne_1 = matric_A[:1,:]
#print(Ligne_1)

produit_sca = Col_1.dot(Ligne_1)
print(pro_sca)

#20
print("Question 20:")

matric_A[3:7,3:7] = np.zeros((4,4))
print(matric_A)

#21
print("Question 21:")

for i in range(len(matric_A)):
    for j in range(len(matric_A)):
        if (matric_A[i][j] == 0):
            print(i,j)



print("hors exo")
print(matric_C)
#pour afficher la 1er ligne
print(len(matric_C[0]))

#pour afficher la 1er colonne
print(matric_C[:,:1])
