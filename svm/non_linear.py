import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def gen_random_sphere(r, R):
    phi = np.random.random() * 360
    return ((R - r)*np.cos(phi)*np.random.random() + r*np.cos(phi), (R - r)*np.sin(phi)*np.random.random() + r*np.sin(phi))

X, Y = [], []
x = np.array([gen_random_sphere(4, 5) for _ in range(1000)])
plt.plot(x[:, 0], x[:, 1], 'b.')
X.extend(x)
Y.extend([0]*1000)


x = np.array([gen_random_sphere(1, 2) for _ in range(500)])
plt.plot(x[:, 0], x[:, 1], 'r.')
X.extend(x)
Y.extend([1]*500)

plt.show()
plt.clf()

X = np.array(X)
y = np.array(Y)


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

color_map = {0: (1, 0, 0), 1: (0, 0, 1)}

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
