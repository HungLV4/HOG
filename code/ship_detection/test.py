import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian

box = np.array([[59, 106], [35,92], [74,26], [98,41],[59, 106]], dtype=np.float64)

img = cv2.imread('VNR20150117_PAN_10.png', 0)

snake = active_contour(img, box, bc='periodic', alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
ax.plot(box[:, 0], box[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
plt.show()