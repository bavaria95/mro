import numpy as np
import pywt
import cv2

def denoise_coefs(A, wavelet='haar', level=None):
    wavelet = pywt.Wavelet(wavelet)

    return pywt.wavedec2(A, wavelet, level=level)

filename = 'image3_gauss'
A = cv2.imread('images/%s.png' % filename)[:, :, 0].astype(float)

for i in range(1, 6):
    coefs = denoise_coefs(A, level=i)

    cv2.imwrite('denoising/%s_approx.png' % i, coefs[0])
    s = coefs[1][0] + coefs[1][1] + coefs[1][2]
    # cv2.imwrite('1.png', coefs[1][0])
    # cv2.imwrite('2.png', coefs[1][1])
    # cv2.imwrite('3.png', coefs[1][2])
    cv2.imwrite('denoising/%s_details.png' % i, s)
