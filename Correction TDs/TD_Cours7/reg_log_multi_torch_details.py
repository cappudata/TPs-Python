import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th

#0) Choix d'une graine aléatoire
np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)


#1) Chargement des données
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target


#2) Affichage des données
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='g', label='2')
plt.legend()
plt.show()



#3) Calcul de N (le nombre d'exemples), calcul de la dimension d (nombre de variables)
N = X.shape[0]
d = X.shape[1]

#4) k est le nombre de classes
k = 3

#5) Ajout d'une première colonne de "1" à la matrice X des entrées
X = np.hstack((np.ones((N,1)),X))

#6) Création d'une matrice de poids avec Pytorch
weights = th.randn(d+1,k)/10

#7)Definition de la fonction softmax
def softmax(z):
    exp = th.exp(z)
    sum = th.sum(exp,1)
    return exp/sum.view(-1,1)

#8) Fonction qui calcule la sortie du modèle pour une matrice de points en entrée
def output(X,weights):
    return softmax(X @ weights)

#9) fonction qui calcule les prédictions (0 ou 1) à partir des sorties du modèle
def prediction(f):
    return th.argmax(f, 1)

#10) Fonction qui calcule le taux d'erreur en comparant les y prédits avec les y réels
def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]


#11) one hot encoding y vector
y_one_hot = np.zeros((N,k))

for i in range(N):
    y_one_hot[i,y[i]] = 1



# 12) Calcul de l'entropie croisée sur l'ensemble du dataset en fonction du jeu de paramètres
def cross_entropy(f_train,y):
    return -(y*th.log(f_train)).sum()/y.shape[0]

# 13) Calcul du gradient de l'erreur
def gradient(X,f_train,y):

    gradient = -(th.transpose(X,0,1) @ (y-f_train))/X.size()[0]
    return gradient


#14) Affichage des frontières de décision
def plot_decision(X,y,weights,device):
    h = 0.05

    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    x_flat = xx.ravel()
    y_flat = yy.ravel()

    x_flat = x_flat.reshape(x_flat.shape[0],1)
    y_flat = y_flat.reshape(y_flat.shape[0], 1)

    X_entry = np.hstack((np.ones((x_flat.shape[0],1)),x_flat,y_flat))

    X_entry = th.from_numpy(X_entry).float().to(device)

    f = output(X_entry, weights)
    y_pred = prediction(f).detach().cpu().numpy()
    preds = y_pred.reshape(xx.shape)


    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.pcolormesh(xx, yy, preds, cmap=cmap_light)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap=cmap_bold)
    plt.show()



#15) Séparation aléatoire du dataset en ensemble d'apprentissage (70%) et de test (30%)
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]
X_train = X[training_idx,:]
y_train = y[training_idx]
y_one_hot_train = y_one_hot[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]
y_one_hot_test = y_one_hot[test_idx]


#16) Spécification du materiel utilisé device = "cpu" pour du calcul CPU, device = "cuda:0" pour du calcul sur le device GPU "cuda:0".
device = "cpu"


#17) Envoi du vecteur de poids sur le device
weights = weights.to(device)



#18) Conversion des données en tenseurs Pytorch et envoi sur le device
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).to(device)
y_one_hot_train = th.from_numpy(y_one_hot_train).float().to(device)

X_test = th.from_numpy(X_test).float().to(device)
y_test = th.from_numpy(y_test).to(device)
y_one_hot_test = th.from_numpy(y_one_hot_test).float().to(device)

#19) Taux d'apprentissage (learning rate)
eta = 0.01


# tqdm permet d'avoir une barre de progression
nb_epochs = 100000

#20) Lancement de l'algorithme d'entraînement
for i in range(nb_epochs):

    f_train = output(X_train,weights)

    weights = weights - eta*gradient(X_train,f_train,y_one_hot_train)

    if(i%10==0):
        y_pred_train = prediction(f_train)
        error_train = error_rate(y_pred_train,y_train)

        f_test = output(X_test, weights)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)

        loss = cross_entropy(f_train,y_one_hot_train)

        print("iter : " + str(i) + " loss " + str(loss.item()) +  " error train : " + str(error_train.item())  + " error test : " + str(error_test.item()))

    if(i%1000==0):
        plot_decision(X, y, weights, device)



