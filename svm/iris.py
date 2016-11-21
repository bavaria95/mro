import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, neighbors
from sklearn.model_selection import train_test_split, cross_val_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C)
rbf_svc = svm.SVC(kernel='rbf', gamma='auto', C=C)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
knn = neighbors.KNeighborsClassifier(1, weights='distance')

print("Linear: %s" % np.average(cross_val_score(svc, X, y, cv=5)))

print("RBF: %s" % np.average(cross_val_score(rbf_svc, X, y, cv=5)))

print("Polynomial: %s" % np.average(cross_val_score(poly_svc, X, y, cv=5)))

print("1NN: %s" % np.average(cross_val_score(knn, X, y, cv=5)))
