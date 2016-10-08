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

def merge_data(featuresTrain, labelsTrain, featuresTest, labelsTest):
    features = np.vstack((featuresTrain, featuresTest))
    labels = np.vstack((labelsTrain, labelsTest))

    return (features, labels)

def shuffle(features, labels):
    ind = np.random.permutation(np.arange(features.shape[0]))
    return (features[ind], labels[ind])

def split_dataset(features, labels, k):
    return (np.array(np.array_split(features, k)),
            np.array(np.array_split(labels, k)))

def cross_validation(features, labels, k):
    features, labels = shuffle(features, labels)
    features_split, labels_split = split_dataset(features, labels, k)

    total_error = 0
    for i in range(k):
        featuresTest = features_split[i]
        labelsTest = labels_split[i]

        chunks_to_merge = np.delete(np.arange(k), i)
        featuresTrain = np.vstack(features_split[chunks_to_merge])
        labelsTrain = np.vstack(labels_split[chunks_to_merge])

        error = test(featuresTrain, labelsTrain, featuresTest, labelsTest)
        total_error += error

    return total_error / k
        

if __name__ == "__main__":
    # yalefaces()
    # print(spambase())
    mat = read_data('datasets/spambase.mat')

    features, labels = merge_data(mat['featuresTrain'], mat['classesTrain'], mat['featuresTest'], mat['classesTest'])
    cross_validation(features, labels, 5)

    
