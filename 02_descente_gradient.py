

if __name__ == '__main__':
    # Fonction à minimiser
    fc = lambda x, y:(3*x**2) + (x*y) + (5*y**2)
    # Détermination des dérivées partielles
    partial_derivative_x = lambda x, y: (6*x) + y
    partial_derivative_y = lambda x, y: (10*y) + x
    # Fixation des variables
    x = 10
    y = -13
    # Taux d'apprentissage
    learning_rate = 0.1
    print("Fc = %s" % (fc(x, y)))
    # Une époque est une période de minimisation
    for epoch in range(0, 20):
        x_gradient = partial_derivative_x(x, y)
        y_gradient = partial_derivative_y(x, y)
        # Application de la descente de gradient
        x = x - learning_rate * x_gradient
        y = y - learning_rate * y_gradient
        # Traçage de la valeur de la fonction
        print("Fc = %s" % (fc(x, y)))
    
    # Affichage des valeurs finales
    print()
    print("x = %s" % x)
    print("y = %s" % y)
