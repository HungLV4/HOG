import numpy as np
from hog import HOGDetector
import cv2 

if __name__ == '__main__':
	image = cv2.imread("test/ship/multi_scale/0000093.png", 0)

	hog = HOGDetector()
	hog.compute(image)