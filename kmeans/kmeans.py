import scipy.io
import numpy as np

def dist(a, b):
    return np.linalg.norm(a - b)

def closest_centroid(point, centroids):
    return np.argmin([dist(point, x) for x in centroids])

def assign_centroids(data, centroids):
    return np.array([closest_centroid(point, centroids) for point in data])

def init_centroids(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

def update_centroids(data, centroids):
    pass

def kmeans(data, k):
    centroids = init_centroids(data, k)

    while True:
        assigned = assign_centroids(data, centroids)
        centroids_upd = np.array([np.mean(data[np.where(assigned==c)[0]], 0) for c in range(centroids.shape[0])])
        if dist(centroids, centroids_upd) < 10e-6:
            break
        centroids = np.matrix.copy(centroids_upd)

    return centroids

def main():
    data = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [4, 4], [4, 5], [5, 4], [5, 5]])

    print(kmeans(data, 1))

main()
