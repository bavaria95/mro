import scipy.io
import numpy as np
import cv2

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

def read_data(filename):
    return scipy.io.loadmat(filename)

def show_similar_faces(data_plain, assigned, k):
    N = 400
    for i in range(k):
        im = np.ones((N, N))
        x, y = 0, 0
        for face in data_plain[np.where(assigned==i)[0]]:
            if x + face.shape[0] >= N:
                x = 0
                y += face.shape[1]

            im[y:y+face.shape[1], x:x+face.shape[0]] = face
            x = x + face.shape[0]

        cv2.imshow(str(i), im)
        cv2.waitKey(0)

def main():
    mat = read_data('datasets/facesYale.mat')
    data = np.concatenate((mat['featuresTrain'], mat['featuresTest']))
    facesTrain = mat['facesTrain']
    facesTrain = np.array([facesTrain[:, :, i] for i in range(facesTrain.shape[2])])
    facesTest = mat['facesTest']
    facesTest = np.array([facesTest[:, :, i] for i in range(facesTest.shape[2])])
    data_plain = np.concatenate((facesTrain, facesTest))
    k = 10

    centroids = kmeans(data, k)
    a = assign_centroids(data, centroids)

    show_similar_faces(data_plain, a, k)

main()
