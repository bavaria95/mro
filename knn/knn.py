import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def read_yale():
    return scipy.io.loadmat('datasets/facesYale.mat')

def dist(a, b):
    return np.linalg.norm(a - b)

def vote(labels, indexes_min):
    return np.argmax(np.bincount(labels[indexes_min].flatten()))

def knn(train, test1, labels, k=1):
    indexes_min = np.argsort([dist(x, test1) for x in train])[ :k]
    return vote(labels, indexes_min)

def scale_feature(x, feature_number, k):
    y = x.copy()
    y[:, feature_number] *= k
    return y

if __name__ == "__main__":
    mat = read_yale()
    x, y = [], []

    for alpha in range(1, 21):
        featuresTrain = scale_feature(mat['featuresTrain'], 9, alpha)
        featuresTest = scale_feature(mat['featuresTest'], 9, alpha)

        correctly_classified = 0
        for i in range(featuresTest.shape[0]):
            predicted = knn(featuresTrain, featuresTest[i], mat['personTrain'], 1)
            actual = mat['personTest'][i]
            if actual == predicted:
                correctly_classified += 1

        print("alpha = %s. acc = %s" % (alpha, float(correctly_classified) / mat['personTest'].shape[0]))
        x.append(alpha)
        y.append(float(correctly_classified) / mat['personTest'].shape[0])

    plt.plot(x, y)
    plt.title('Dependence of accuracy from scaling the last feature')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('accuracy')
    plt.show()
