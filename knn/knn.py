import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def read_data(filename):
    return scipy.io.loadmat(filename)

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

def yalefaces():
    mat = read_data('datasets/facesYale.mat')
    x, y = [], []

    for alpha in range(1, 21):
        featuresTrain = scale_feature(mat['featuresTrain'], 9, alpha)
        featuresTest = scale_feature(mat['featuresTest'], 9, alpha)

        misclassified = 0
        for i in range(featuresTest.shape[0]):
            predicted = knn(featuresTrain, featuresTest[i], mat['personTrain'])
            actual = mat['personTest'][i]
            if actual == predicted:
                misclassified += 1

        print("alpha = %s. error = %s" % (alpha, 100.0 - 100 * float(correctly_classified) / mat['personTest'].shape[0]))
        x.append(alpha)
        y.append(100*float(misclassified) / mat['personTest'].shape[0])

    plt.plot(x, y)
    plt.title('Dependence of error from scaling the last feature')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('error, [%]')
    plt.show()

def spambase():
    mat = read_data('datasets/spambase.mat')

    misclassified = 0
    for i in range(mat['featuresTest'].shape[0]):
        predicted = knn(mat['featuresTrain'], mat['featuresTest'][i], mat['classesTrain'])
        actual = mat['classesTest'][i]
        if actual != predicted:
            misclassified += 1

    print(100.0 * float(misclassified) / mat['classesTest'].shape[0])


if __name__ == "__main__":
    # yalefaces()
    spambase()
    
