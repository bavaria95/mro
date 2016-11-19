import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets

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
plt.plot(X[y_2, 0], X[y_2, 1], 'go')

plt.show()
