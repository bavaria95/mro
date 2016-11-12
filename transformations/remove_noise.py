from numpy import *
import scipy
import pywt
import cv2

filename = 'image4_gauss'
A = cv2.imread('images/%s.png' % filename)[:, :, 0].astype(float)

wavelet = pywt.Wavelet('haar')
 
coefs = pywt.wavedec2(A, wavelet)

threshold = 2*sqrt(2*log2(A.size))
coefs = map(lambda x: pywt.threshold(x, threshold), coefs)
A = pywt.waverec2(coefs, wavelet)

cv2.imwrite('%s_denoised.png' % filename, A)
