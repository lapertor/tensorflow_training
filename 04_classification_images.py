import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# On fait ça en one-hot (vecteur de 0 ou 1 pour chaque valeur possible) plutot qu'avoir la 
# valeur directement, car dans ce cas là c'est plus pratique
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Détermination des entrées du graph
tf_features = tf.placeholder(tf.float32, [None, 784])
tf_targets = tf.placeholder(tf.float32, [None, 10])

# Variables du réseau de neurones (poids et biais)
w1 = tf.Variable(tf.random_normal([784, 10]))
b1 = tf.Variable(tf.zeros([10]))

# Opérations

# Pré-activations
z1 = tf.matmul(tf_features, w1) + b1

# Activation avec Softmax (plus sigmoid)
softmax = tf.nn.softmax(z1)

# Erreur (avec données one-hot et sortie softmax)
error = tf.nn.softmax_cross_entropy_with_logits(labels=tf_targets, logits=z1)

# Entrainement
train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(error)

# Mesure de la réussite de l'entrainement (accuracy)
correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(tf_targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 1000
    for e in range(epochs):
        # Un batch est un sous-ensemblme de la dataset, qui peut etre trop grosse souvent
        batch_features, batch_targets = mnist.train.next_batch(100)
        sess.run(train, feed_dict={
            tf_features: batch_features, 
            tf_targets: batch_targets
        })

    acc = sess.run(accuracy, feed_dict={
        tf_features: mnist.test.images,
        tf_targets: mnist.test.labels
    }) 

    py = sess.run(softmax, feed_dict={
        tf_features: [mnist.test.images[0]],
        tf_targets: [mnist.test.labels[0]]
    }) 
    cls = np.argmax(py)
    print("softmax =", py)
    print("cls =", cls)

    print("Accuracy = ", acc)

    

    sess.run(train, feed_dict={
        tf_features: [mnist.train.images[0]],
        tf_targets: [mnist.train.labels[0]]
    })

