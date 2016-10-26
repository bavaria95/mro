import imageio
import glob
import cv2

images = []
filenames = sorted(glob.glob("*.png"), key=lambda x: int(x.split('.')[0]))
for fn in filenames:
    for i in range(5):
        img = cv2.imread(fn)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, fn.split('.')[0], (50, 50), font, 2, (0, 0, 0), 2)
        images.append(img)

imageio.mimsave('movie.gif', images)
