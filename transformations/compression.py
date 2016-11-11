import numpy as np
import pywt
import cv2
import copy

def diff(A, B):
    return np.linalg.norm(A - B)

def compress_image(A, wavelet='haar'):
    for i in range(A.shape[0]):
        x = A[i].astype(float)
        n = x.shape[0]
        for _ in range(int(np.log2(n))):
            cA, cD = pywt.dwt(x[ :n], wavelet)
            x[0:n/2] = cA
            x[n/2:n] = cD
            n /= 2

        A[i] = x

    for i in range(A.shape[1]):
        x = A[:, i]
        n = x.shape[0]
        for _ in range(int(np.log2(n))):
            cA, cD = pywt.dwt(x[ :n], wavelet)
            x[0:n/2] = cA
            x[n/2:n] = cD
            n /= 2

        A[:, i] = x

def decompress_image(A, wavelet='haar'):
    for i in range(A.shape[0]):
        x = A[i].astype(float)
        n = 1
        for _ in range(int(np.log2(x.shape))):
            cA = x[0:n]
            cD = x[n:2*n]
            x[ :2*n] = pywt.idwt(cA, cD, wavelet)
            n *= 2

        A[i] = x

    for i in range(A.shape[1]):
        x = A[:, i].astype(float)
        n = 1
        for _ in range(int(np.log2(x.shape))):
            cA = x[0:n]
            cD = x[n:2*n]
            x[ :2*n] = pywt.idwt(cA, cD, wavelet)
            n *= 2

        A[:, i] = x

eps = 5

filename = 'image1'
A = cv2.imread('images/%s.png' % filename)[:, :, 0].astype(float)
initial_nonzeros = np.where(A!=0)[0].shape[0]

compress_image(A)

A[np.where(np.abs(A) < eps)] = 0

approx_nonzeros = np.where(A!=0)[0].shape[0]

print(float(initial_nonzeros) / approx_nonzeros)

decompress_image(A)

cv2.imwrite('%s_compressed.png' % filename, A.astype(np.uint8))

# cv2.imshow('img', A.astype(np.uint8))
# cv2.waitKey(0)
