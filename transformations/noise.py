import numpy as np
import os
import cv2

def noisy(noise_typ, image):
   if noise_typ == "gauss":
      row,col= image.shape
      mean = 0
      var = 400
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.05
      out = np.copy(image)

      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out

filename = 'image4'
A = cv2.imread('images/%s.png' % filename)[:, :, 0].astype(float)

A_sp = noisy('s&p', A)
A_g = noisy('gauss', A)

cv2.imwrite('images/%s_sp.png' % filename, A_sp)
cv2.imwrite('images/%s_gauss.png' % filename, A_g)
