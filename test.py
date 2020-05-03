import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np


def plot2D(data):
    plt.title('Dispercao embedings')
    plt.xlabel('x')
    plt.ylabel('y')
    labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle', 'boot']
    aux = []
    for name, hex in matplotlib.colors.cnames.items():
        aux.append(hex)
    colors = random.sample(aux, 10)
    for x, y, t in data:
        plt.scatter(x, y, c=colors[int(t)])

    plt.legend(labels)
    plt.savefig('111111.png')
    plt.close()


if __name__ == '__main__':
    X = np.random.uniform(size=(100, 2))
    n, m = X.shape  # for generality
    X0 = np.ones((n))
    print(X.shape)
    print(X0.shape)
    v = X0.shape
    print(v[0])
    X0 = X0.reshape((v[0], 1))
    print(X0.shape)
    Xnew = np.hstack((X, X0))
#    print(Xnew)
