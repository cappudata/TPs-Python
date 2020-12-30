# 1)
x = int(input("Entrez un nombre"))

while x < 0:
    x = int(input("Entrez un nombre"))

if x%2 == 0:
    print("Pair")
else:
    print("Impair")

#en une ligne avec une expression ternaire

print("Pair") if int(input("Entrez un nombre")) %2==0 else print("Impair")

#2)
n = 100
cpt = 0

while n%2==0:
    cpt+=1
    n/=2

print(cpt)

# 3)
sum = 0
for i in range(15):
    sum += i
print(sum)

# 4)
n = 12
cpt = 0

for i in range(1,7):
    for j in range(1,7):
        if i+j == n:
            cpt +=1

#5)
def minMaxMoy(liste):

    min = liste[0]
    max = liste[0]
    average = 0

    for elt in liste:

        if elt < min:
            min = elt

        if elt > max:
            max = elt

        average += elt

    average /= len(liste)

    return min, max, average

liste = [10, 12, 18, 14]

min, max, average = minMaxMoy(liste)

fichier = open("test.txt" , "a")

fichier.write("min : " + str(min) + "\n")
fichier.write("max : " +  str(max) + "\n")
fichier.write("average : " + str(average) + "\n")
fichier.close()










