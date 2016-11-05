import numpy as np
import pywt
import cv2

A = cv2.imread('images/image3.png')[:, :, 0].astype(float)
print(A.shape)

for i in range(A.shape[0]):
    x = A[i].astype(float)
    n = x.shape[0]
    for _ in range(3):
        cA, cD = pywt.dwt(x[ :n], 'haar')
        x[0:n/2] = cA
        x[n/2:n] = cD
        n /= 2
        # print(x)
        print(np.where(abs(x) < 5)[0].shape)
    print

    A[i] = x

print(A)



