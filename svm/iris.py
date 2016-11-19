import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma='auto', C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)

Z = svc.predict(X_test)
print("Linear: %s" % (sum(map(int, Z == y_test))/(float(y_test.shape[0]))))

Z = rbf_svc.predict(X_test)
print("RBF: %s" % (sum(map(int, Z == y_test))/(float(y_test.shape[0]))))

Z = poly_svc.predict(X_test)
print("Polynomial: %s" % (sum(map(int, Z == y_test))/(float(y_test.shape[0]))))
