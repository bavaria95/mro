import numpy as np
import scipy.io

def read_yale():
    return scipy.io.loadmat('datasets/facesYale.mat')

def dist(a, b):
    return np.linalg.norm(a - b)

def vote(labels, indexes_min):
    return np.argmax(np.bincount(labels[indexes_min].flatten()))

def knn(train, test1, labels, k=1):
    indexes_min = np.argsort([dist(x, test1) for x in train])[ :k]
    return vote(labels, indexes_min)

if __name__ == "__main__":
    mat = read_yale()
    correctly_classified = 0
    for i in range(mat['featuresTest'].shape[0]):
        predicted = knn(mat['featuresTrain'], mat['featuresTest'][i], mat['personTrain'], 1)
        actual = mat['personTest'][i]
        if actual == predicted:
            correctly_classified += 1

    print(float(correctly_classified) / mat['personTest'].shape[0])

