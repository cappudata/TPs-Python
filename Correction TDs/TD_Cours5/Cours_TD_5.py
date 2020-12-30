import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets


#1) Chargement du dataset Iris
iris = datasets.load_iris()

#2) Selection des deux premiers attributs et remplacement de la classe 2 par la classe 1
X = iris.data[:,:2]
y = iris.target
y[y==2] = 1

#3) Affichage du dataset
plt.scatter(X[y==0,0],X[y==0,1],color="b")
plt.scatter(X[y==1,0],X[y==1,1],color="r")
plt.show()

N = X.shape[0]
d = X.shape[1]

#3) Création d'un vecteur de poids
weights = np.random.randn(d+1)
print(weights)

#4) Ajout d'une première colonne de "1" à la matrice X des entrées
X = np.hstack((np.ones((N,1)),X))

#5) Calcul du produit scalaire entre la première entrée du dataset et le vecteur de poids
first_entry = X[0,:]
scalar_product = (weights * first_entry).sum()

#6) Fonction sigmoid
def sigmoid(z):
    return 1 / (1+np.exp(-z))

#7) Calcul de la sortie du modèle pour la première entrée
f = sigmoid(scalar_product)
print("f " + str(f))

#8) Calcul de la prédiction faite par le modèle pour la première entrée
pred = f.round()
print("pred " + str(pred))

#9) Fonction qui calcule la sortie du modèle pour une matrice de points en entrée
#a) Version naive avec une boucle
def output(X,weights):
    out = np.zeros((N,1))
    for i in range(N):
        out[i] = sigmoid((weights * X[i,:]).sum())
    return out

print(output(X,weights))

#b) Version avec un calcul matriciel
def output(X,weights):

    return sigmoid(np.dot(X,weights))


#10) fonction qui calcule les prédictions (0 ou 1) à partir des sorties du modèle
def prediction(f):

    return f.round()

#11) Fonction qui calcule le taux d'erreur en comparant le y prédit avec le y réel
def error_rate(y_pred,y):
    return (y_pred!=y).mean()


#12) Affichage de la frontière de décision
def plot_boundary_decision(X,weights):

    x1_plot = np.linspace(0,10)
    x2_plot = (-weights[0] - weights[1]*x1_plot)/weights[2]

    plt.scatter(X[y==0,1],X[y==0,2],color="b")
    plt.scatter(X[y==1,1],X[y==1,2],color="r")

    plt.plot(x1_plot,x2_plot)

    plt.show()


#13) Affichage d'une carte des décisions prises par le modèle
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])

def plot_decision(X,y,weights):

    h = 0.1

    xx1, xx2 = np.meshgrid(np.arange(0,10,h),np.arange(0,10,h))

    # print(xx1)
    # print(xx2)

    xx1_flat = xx1.reshape(xx1.shape[0]**2,1)
    xx2_flat = xx2.reshape(xx2.shape[0]**2,1)


    X_entry = np.hstack((np.ones((xx1_flat.shape[0],1)),xx1_flat,xx2_flat))

    # print(X_entry)

    f = output(X_entry,weights)
    y_pred = prediction(f)

    yy = y_pred.reshape(xx1.shape[0],xx1.shape[1])

    plt.pcolormesh(xx1,xx2,yy,cmap=cmap_light)

    plt.scatter(X[y==0,1],X[y==0,2],color="r")
    plt.scatter(X[y==1,1],X[y==1,2],color="g")

    plt.show()



#14) Calcul de la binary cross entropy entre le vecteur de sortie du modèle et le vecteur des targets.
def binary_cross_entropy(f,y):

    return - (y*np.log(f)+ (1-y)*np.log(1-f)).mean()


#15) Calcul du gradient de l'erreur par rapport aux paramètres du modèle
#a) Version avec une boucle
def gradient(f,y,X):

    grad = np.zeros((d+1,1))

    for j in range(0,d+1):

        grad[j] = -((y-f)*X[:,j]).mean()

        print(-((y-f)*X[:,j]).mean())

    return grad

#b) Version avec un calcul matriciel
def gradient_dot(f,y,X):

    grad = -np.dot(np.transpose(X),(y-f))/X.shape[0]

    return grad






#16) Séparation aléatoire du dataset en ensemble d'apprentissage (70%) et de test (30%)

indices = np.random.permutation(X.shape[0])

training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]


X_train = X[training_idx,:]
y_train = y[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]




#17) Taux d'apprentissage (learning rate)
eta = 0.01


#18) Apprentissage du modèle et calcul de la performance tous les 100 itérations
nb_epochs = 10000
for i in range(nb_epochs):

    f_train = output(X_train,weights)
    y_pred_train = prediction(f_train)

    grad = gradient_dot(f_train,y_train,X_train)

    weights = weights - eta*grad

    if(i%100==0):

        error_train = error_rate(y_pred_train,y_train)
        loss = binary_cross_entropy(f_train,y_pred_train)

        f_test = output(X_test, weights)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)

        plot_decision(X_train, y_train, weights)

        print("iter : " + str(i) +  " error train : " + str(error_train) + " loss " + str(loss) + " error test : " + str(error_test))

#19) Affichage des paramètres appris du modèle
print("weights")
print(weights)




#20) Lancement du modèle avec la librairie scikit-learn et affichage des résultats
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred_test = model.predict(X_test)

error_test = error_rate(y_pred_test, y_test)

print("error_test")
print(error_test)

print("Valeurs des poids du model avec scikitlearn")
print(model.intercept_, model.coef_)










