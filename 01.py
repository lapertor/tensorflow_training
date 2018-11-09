"""
Avec ce programme, on cherche à différencier les personnes malades
des personnes saines à partir de leur taux de globules blancs et 
de globules rouges

DATASET = [features, targets]
features = [taux_globules_blanc_1   taux_globules_rouges_1
.                   ...                     ...           ]
targets = [0 pour malade, 1 pour positif
.                       ...             ]

On va utiliser un neurone :
    - Prends une ligne des features
    - Leur attribut des poids et un biais
    - Fait une pré-activation
    - Normalise la pré-activation grâce à l'activation
    - Sort l'activation

On détermine ensuite pour chaque personne une erreur (cost) sur la 
sortie du neurone en comparant avec les targets (résultats attendus)

À partir de ce cost, on ajuste les poids et biais en faisant une 
moyenne de leurs gradients pour chaque personne
"""

import numpy as np
import matplotlib.pyplot as plt


def init_variables():
    """
        Initialise les variables du modèle (poids et biais)
    """
    weights = np.random.normal(size=2)
    bias = 0
    return weights, bias

def get_dataset():
    """
        Procédure générant une dataset aléatoire
    """
    # Nombre de lignes par classe
    row_per_class = 100
    # Génération des lignes sous forme de matrice
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])

    sick2 = np.random.randn(row_per_class, 2) + np.array([2, -2])
    healthy2 = np.random.randn(row_per_class, 2) + np.array([-2, 2])

    features = np.vstack([sick, sick2, healthy, healthy2])
    targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    return features, targets

def pre_activation(features, weights, bias):
    """
        Calcule la pré-activation
    """
    return np.dot(features, weights) + bias

def activation(z):
    """
        Calcul l'activation
    """
    return 1 / (1 + np.exp(-z))

def derivative_activation(z):
    """
    """
    return activation(z) * (1 - activation(z))

def predict(features, weights, bias):
    """
    """
    z = pre_activation(features, weights, bias)
    y = activation(z)
    return np.round(y)

def cost(predictions, targets):
    """
    """
    return np.mean( (predictions - targets)**2 )

def train(features, targets, weights, bias):
    """
    """

    epochs = 100
    learning_rate = 0.1

    # Affichage de la précision
    predictions = predict(features, weights, bias)
    print("Accuracy = ", np.mean(predictions == targets))

    # Affichage des points
    plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()

    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print("Cost = %s" % cost(predictions, targets))
        # Initialisation du gradient
        weights_gradients = np.zeros(weights.shape)
        bias_gradient = 0
        # Parcours des lignes
        for feature, target in zip(features, targets):
            # calcul de la prédiction
            z = pre_activation(feature, weights, bias)
            y = activation(z)
            # Mise à jour des gradients
            weights_gradients += (y - target) * derivative_activation(z) * feature
            bias_gradient += (y - target) * derivative_activation(z)
        # mise à jour des variables
        weights = weights - learning_rate * weights_gradients
        bias = bias - learning_rate * bias_gradient
    
    predictions = predict(features, weights, bias)
    print("Accuracy = ", np.mean(predictions == targets))




if __name__ == '__main__':
    # Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    train(features, targets, weights, bias)
    pass