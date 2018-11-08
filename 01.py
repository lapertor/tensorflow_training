import numpy as np

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
    row_per_class = 5
    # Génération des lignes sous forme de matrice
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])

    features = np.vstack([sick, healthy])
    targets = np.concatenate((np.zeros(row_per_class), np.zeros(row_per_class) + 1))

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

if __name__ == '__main__':
    # Dataset
    features, targets = get_dataset()
    # Variables
    weights, bias = init_variables()
    # Calcul de la pré-activation
    z = pre_activation(features, weights, bias)
    # Calcul de l'activationn
    a = activation(z)
    pass