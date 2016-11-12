import numpy as np
from scipy import fftpack
import cv2

def diff(A, B):
    return np.linalg.norm(A - B)

def get_2D_dct(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

def get_2d_idct(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

filename = 'image1'
A = cv2.imread('images/%s.png' % filename)[:, :, 0].astype(float)
initial_nonzeros = np.where(A!=0)[0].shape[0]

dct = get_2D_dct(A)

ii = 300
dct[ii: , : ] = 0
dct[ : , ii: ] = 0

approx_nonzeros = np.where(dct!=0)[0].shape[0]
print(float(initial_nonzeros) / approx_nonzeros)

r_img = get_2d_idct(dct);

print(diff(A, r_img))

cv2.imshow('img', r_img.astype(np.uint8))
cv2.waitKey(0)
