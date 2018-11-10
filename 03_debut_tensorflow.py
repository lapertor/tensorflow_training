"""
On va maintenant utiliser Tensorflow
Tensorflow utilise un système de graph, dans lequel on va définir des opérations,
puis exécuter ces opérations.
Le graph a besoin qu'on lui indique ses entrées (==> placeholder) et ses sorties
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    # plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    # plt.show()

    # Un placeholder indique au graph les entrées dont il va avoir besoin. 
    # - None peut etre remplacé par 400
    tf_features = tf.placeholder(tf.float32, shape=[None, 2])
    tf_targets = tf.placeholder(tf.float32, shape=[None, 1])

    # Impression des features dans le graph de la session
    #print(sess.run(tf_features, feed_dict={
    #    tf_features: features
    #}))

    # Déclaration puis initialisation des variables
    # Premières liaisons entrée-neurones du hidden layout
    w1 = tf.Variable(tf.random_normal([2, 3]))
    b1 = tf.Variable(tf.zeros([3]))


    # Déclaration de la préactivation puis de l'activation des premiers neurones du hidden layout
    z1 = tf.matmul(tf_features, w1) + b1
    a1 = tf.nn.sigmoid(z1)

    # Neurone de sortie : variables
    w2 = tf.Variable(tf.random_normal([3, 1]))
    b2 = tf.Variable(tf.zeros([1]))

    # Préactivation et activation du neurone de sortie
    z2 = tf.matmul(a1, w2) + b2
    py = tf.nn.sigmoid(z2)

    # Calcul du cout (erreur) de la prédiction par rapport à la valeur réelle
    cost = tf.reduce_mean(tf.square(py - tf_targets))

    # Détermination de la précision de la prédiction effectuée
    correct_prediction = tf.equal(tf.round(py), tf_targets)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Optimisation des poids et biais en utilisant une descente de gradient
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)


    # Une session contient le graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for e in range(10000):
        sess.run(train, feed_dict={
            tf_features: features,
            tf_targets: targets
        })

        if e % 500 == 0:
            print("accuracy = ", sess.run(accuracy, feed_dict={
                tf_features: features,
                tf_targets: targets
            }))