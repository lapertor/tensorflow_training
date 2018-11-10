import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_dataset():
    """
        Procédure générant une dataset aléatoire
    """
    # Nombre de lignes par classe
    row_per_class = 100
    # Génération des lignes sous forme de matrice
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    sick2 = np.random.randn(row_per_class, 2) + np.array([2, 2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, -2])
    healthy2 = np.random.randn(row_per_class, 2) + np.array([-2, 2])

    features = np.vstack([sick, sick2, healthy, healthy2])
    targets = np.concatenate((np.zeros(row_per_class * 2), np.zeros(row_per_class * 2) + 1))

    targets = targets.reshape(-1, 1)

    return features, targets
    
if __name__ == '__main__':
    # Dataset
    features, targets = get_dataset()
    # Affichage des points
    plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.show()