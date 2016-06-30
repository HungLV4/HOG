from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import cv2
import imutils

import os.path
import sys

CLF_FILE = "genfiles/digits_cls.pkl"

def train():
	dataset = datasets.fetch_mldata("MNIST Original")
	images = np.array(dataset.data, 'int16')
	labels = np.array(dataset.target, 'int')

	list_hog_fd = []
	for im in images:
		fd = hog(im.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)
	hog_features = np.array(list_hog_fd, 'float64')

	clf = LinearSVC()
	clf.fit(hog_features, labels)
	joblib.dump(clf, CLF_FILE, compress=3)

def test(imageFile):
	# load classifier
	clf = joblib.load(CLF_FILE)

	# read the input image
	im = cv2.imread(imageFile)

	# Convert to grayscale and apply Gaussian filtering
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

	# Threshold the image
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

	ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]

	for rect in rects:
		cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
		
		leng = int(rect[3] * 1.6)
		pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
		pt1 = pt1 if pt1 > 0 else 0

		pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
		pt2 = pt2 if pt2 > 0 else 0

		roi = im_th[pt1 : pt1 + leng, pt2 : pt2 + leng]
		
		roi = cv2.resize(roi, (28, 28))
		roi = cv2.dilate(roi, (3, 3))

		roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
		cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

	cv2.imshow("Resulting Image with Rectangular ROIs", im)
	cv2.waitKey()

if __name__ == '__main__':
	if not os.path.isfile(CLF_FILE):
		train()

	# TESTING
	test(sys.argv[1])
	