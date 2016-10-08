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

def test(featuresTrain, labelsTrain, featuresTest, labelsTest, k=1):
    misclassified = 0
    for i in range(featuresTest.shape[0]):
        predicted = knn(featuresTrain, featuresTest[i], labelsTrain, k)
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
        
def spambase_cross_validation():
    mat = read_data('datasets/spambase.mat')

    for i in range(10):
        features, labels = merge_data(mat['featuresTrain'], mat['classesTrain'], mat['featuresTest'], mat['classesTest'])
        print(cross_validation(features, labels, 5))

def cubes():
    mat = read_data('datasets/multiDimHypercubes.mat')
    maxDim = mat['maxDim'][0][0]

    errors = []
    for k in range(maxDim):
        error = test(mat['featuresTrain'][0][k], mat['classesTrain'][0][k],
                     mat['featuresTest'][0][k], mat['classesTest'][0][k])

        errors.append(error)

    errors = np.array(errors) * 100

    plt.plot(range(1, maxDim + 1), errors)
    plt.title('Dependence of error from number of hypercube dimensions')
    plt.xlabel('i')
    plt.ylabel('error, [%]')
    plt.show()

def cubes_2():
    mat = read_data('datasets/multiDimHypercubes.mat')
    maxDim = mat['maxDim'][0][0]

    k = 0
    featuresTrain = mat['featuresTrain'][0][k]
    classesTrain = mat['classesTrain'][0][k]
    featuresTest = mat['featuresTest'][0][k]
    classesTest = mat['classesTest'][0][k]

    for i in range(featuresTest.shape[0]):
        min_dist_to_the_same_class = []
        min_dist_to_the_diff_class = []

        label = classesTest[i]

        same_class_features = featuresTrain[np.where(classesTrain == label)]
        min_dist = 10**8
        for train_ex in same_class_features:
            min_dist = min(min_dist, dist(train_ex, featuresTest[i]))
        min_dist_to_the_same_class.append(min_dist)

        diff_class_features = featuresTrain[np.where(classesTrain != label)]
        min_dist = 10**8
        for train_ex in diff_class_features:
            min_dist = min(min_dist, dist(train_ex, featuresTest[i]))
        min_dist_to_the_diff_class.append(min_dist)

    print(sum(min_dist_to_the_same_class) / len(min_dist_to_the_same_class))
    print(sum(min_dist_to_the_diff_class) / len(min_dist_to_the_diff_class))




if __name__ == "__main__":
    # yalefaces()
    # print(spambase())
    # spambase_cross_validation()
    # cubes()
    cubes_2()

