import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
X = iris.data
y = iris.target

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

y_0 = np.where(y==0)
plt.plot(X[y_0, 0], X[y_0, 1], 'ro')

y_1 = np.where(y==1)
plt.plot(X[y_1, 0], X[y_1, 1], 'bo')

y_2 = np.where(y==2)
plt.plot(X[y_2, 0], X[y_2, 1], 'yo')
plt.show()

# step size in the mesh
h = .02

lin_svc = (svm.SVC(kernel='linear').fit(X, y), y)
poly_svc = (svm.SVC(kernel='poly').fit(X, y), y)
rbf_svc = (svm.SVC(kernel='rbf').fit(X, y), y)
sig_svc = (svm.SVC(kernel='sigmoid').fit(X, y), y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['Linear kernel',
          'Polynomial kernel',
          'RBF kernel',
          'Sigmoid kernel']

color_map = {0: (1, 0, 0), 1: (0, 0, 1), 2: (0.8, 0.6, 0)}

for i, (clf, y_train) in enumerate((lin_svc, poly_svc, rbf_svc, sig_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, cmap=plt.cm.Paired)

    plt.title(titles[i])

plt.show()
