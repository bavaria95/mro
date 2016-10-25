import colorsys
import numpy as np
import cv2

n = 30

x = np.zeros((200, 850, 3))
h = 0

for i in range(0, 20*n, 20):
    h += 360.0/n
    l = np.random.random()*10 + 50
    s = np.random.random()*10 + 90

    r, g, b = colorsys.hls_to_rgb(h / 360.0, l/100.0, s/100.0)

    x[0:20, i:i+20] = np.array([r, g, b])

cv2.imshow('qwe', x)
cv2.waitKey(0)

