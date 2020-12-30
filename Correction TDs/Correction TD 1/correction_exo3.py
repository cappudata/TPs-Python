
class Vecteur2D:

    def __init__(self,x=0,y=0):

        self.x = x
        self.y = y

    def getDescription(self):

        return "x : " + str(self.x) + " y : " + str(self.y)

    def produit_scalaire(self,vecteur):

        return self.x * vecteur.x + self.y*vecteur.y


vecteur1 = Vecteur2D(2,3)
vecteur2 = Vecteur2D(5,4)

print("Produit scalaire " + str(vecteur1.produit_scalaire(vecteur2)))


