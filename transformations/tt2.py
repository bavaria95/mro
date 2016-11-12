import numpy as np
from scipy import fftpack
import cv2
import matplotlib.pyplot as plt

def diff(A, B):
    return np.linalg.norm(A - B)

def get_2D_dct(img):
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

def get_2d_idct(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

filename = 'image4'
A = cv2.imread('images/%s.png' % filename)[:, :, 0].astype(float)
initial_nonzeros = np.where(A!=0)[0].shape[0]

dct = get_2D_dct(A)

e, c = [], []

for ii in range(200, A.shape[0]):
    dct_copy = np.copy(dct)
    dct_copy[ii: , : ] = 0
    dct_copy[ : , ii: ] = 0

    approx_nonzeros = np.where(dct_copy!=0)[0].shape[0]
    c.append(float(initial_nonzeros) / approx_nonzeros)
    r_img = get_2d_idct(dct_copy);

    e.append(diff(A, r_img))

plt.xkcd()
plt.plot(c, e, '-')
plt.xlabel('compression')
plt.ylabel('error')
plt.show()
