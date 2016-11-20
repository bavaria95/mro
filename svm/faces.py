import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def scale_10_features(X, factor=20):
    x = X.copy()
    x[: 10] *= factor
    return x

lfw_people = fetch_lfw_people(data_home='/home/bavaria/Coding/mro/svm', min_faces_per_person=70)
n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]


n_components = 150

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)

eigenfaces = pca.components_.reshape((n_components, h, w))
X_pca = pca.transform(X)


faces_titles = ["face %d" % i for i in range(X.shape[0])]

plot_gallery(X, faces_titles, h, w)
plt.show()

X_res = pca.inverse_transform(X_pca)
plot_gallery(X_res, faces_titles, h, w)
plt.show()

X_pca_sc = map(scale_10_features, X_pca)
X_res_sc = pca.inverse_transform(X_pca_sc)
plot_gallery(X_res_sc, faces_titles, h, w)
plt.show()
