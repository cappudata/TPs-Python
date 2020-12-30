
# 1)
class Animal():

    def __init__(self, taille=10):

        self.taille = taille

    def mange(self):

        print("je mange")


class Poisson(Animal):

    def nage(self):

        print("je nage")


class Volant():

    def vole(self):

        print("je vole")

class Oiseau(Poisson, Volant):

    def vole(self):

        print("Je vole comme un oiseau")


oiseau = Oiseau(50)


oiseau.vole()


