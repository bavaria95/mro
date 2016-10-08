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

def test(featuresTrain, labelsTrain, featuresTest, labelsTest):
    misclassified = 0
    for i in range(featuresTest.shape[0]):
        predicted = knn(featuresTrain, featuresTest[i], labelsTrain)
        actual = labelsTest[i]
        if actual != predicted:
            misclassified += 1

    return float(misclassified) / labelsTest.shape[0]

def yalefaces():
    mat = read_data('datasets/facesYale.mat')
    x, y = [], []

    for alpha in range(1, 21):
        featuresTrain = scale_feature(mat['featuresTrain'], 9, alpha)
        featuresTest = scale_feature(mat['featuresTest'], 9, alpha)

        error = test(featuresTrain, mat['personTrain'], featuresTest, mat['personTest'])

        print("alpha = %s. error = %s" % (alpha, error))
        x.append(alpha)
        y.append(error * 100)

    plt.plot(x, y)
    plt.title('Dependence of error from scaling the last feature')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('error, [%]')
    plt.show()

def spambase():
    mat = read_data('datasets/spambase.mat')

    return test(mat['featuresTrain'], mat['classesTrain'], mat['featuresTest'], mat['classesTest'])

if __name__ == "__main__":
    # yalefaces()
    print(spambase())
    
