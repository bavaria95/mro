import numpy as np
import pywt
import cv2

def denoise(A, wavelet='haar'):
    wavelet = pywt.Wavelet(wavelet)

    coefs = pywt.wavedec2(A, wavelet)

    threshold = 3*np.sqrt(2*np.log2(A.size))
    coefs = map(lambda x: pywt.threshold(x, threshold), coefs)
    A = pywt.waverec2(coefs, wavelet)

    return A

filename = 'image1_sp'
A = cv2.imread('images/%s.png' % filename)[:, :, 0].astype(float)
A = denoise(A, 'haar')
cv2.imwrite('%s_denoised.png' % filename, A)
