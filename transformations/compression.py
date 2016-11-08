import numpy as np
import pywt
import cv2

eps = 5

def compress_image(A, levels=3):
    for i in range(A.shape[0]):
        x = A[i].astype(float)
        n = x.shape[0]
        for _ in range(levels):
            cA, cD = pywt.dwt(x[ :n], 'haar')
            x[0:n/2] = cA
            x[n/2:n] = cD
            n /= 2

        A[i] = x


A = cv2.imread('images/image3.png')[:, :, 0].astype(float)
initial_nonzeros = np.where(A!=0)[0].shape[0]

compress_image(A)

A[np.where(np.abs(A) < eps)] = 0

approx_nonzeros = np.where(A!=0)[0].shape[0]
print(initial_nonzeros)
print(approx_nonzeros)
print(float(initial_nonzeros) / approx_nonzeros)
