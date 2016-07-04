
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import csv
import math
import numpy as np
import cv2
import imutils
from imutils.object_detection import non_max_suppression

import os.path
import sys
import glob

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

HOG_CLF_FILE = "genfiles/ships_hog_clf.pkl"
TRAIN_LABEL_FILE = "genfiles/ships_label.csv"
TRAIN_FEATURES_FILE = "genfiles/ships_feature.dat"

FILEPATH_PREFIX = "../../../../temp/CT_OCEAN/"

def getAllFilesInDirectory(dir):
	return glob.glob(dir)

def trainHOG():
	trainPositiveFiles = getAllFilesInDirectory("train/ship/pos/64x128/*.png")
	trainNegativeFiles = getAllFilesInDirectory("train/ship/neg/64x128/*.png")

	labels = np.array([-1 for i in range(len(trainNegativeFiles))] + \
						[1 for i in range(len(trainPositiveFiles))])

	list_hog_fd = []
	for filepath in (trainNegativeFiles + trainPositiveFiles):
		im = cv2.imread(filepath, 0)
		fd = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)

		list_hog_fd.append(fd)
	hog_features = np.array(list_hog_fd, 'float64')

	clf = SVC(kernel = 'linear')
	clf.fit(hog_features, labels)
	joblib.dump(clf, HOG_CLF_FILE, compress=3)

def detect():
	# load classifier
	clf = joblib.load(HOG_CLF_FILE)
	
	testFiles = getAllFilesInDirectory("test/ship/64x128/*.png")
	for filepath in testFiles:
		im = cv2.imread(filepath, 0)
		hog_fd = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
		nbr = clf.predict(np.array([hog_fd], 'float64'))

		print filepath, nbr[0] == 1

def detectScene(filename, winStride = (4, 4)):
	# read image
	filepath = "test/ship/multi_scale/" + filename
	image = cv2.imread(filepath, 0)

	# load classifier
	clf = joblib.load(HOG_CLF_FILE)

	height, width = image.shape

	# detect ships in image
	positions = []
	for i in xrange(0, height - 128 - winStride[0], winStride[0]):
		print float(i) / height * 100
		for j in xrange(0, width - 64 - winStride[0], winStride[1]):
			hog_fd = hog(image[i : i + 128, j : j + 64], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)
			nbr = clf.predict(np.array([hog_fd], 'float64'))
			if nbr[0] == 1:
				positions.append((j, i))
	
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[j, i, j + 64, i + 128] for (j, i) in positions])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.5)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 1)

	cv2.imwrite("results/ship/multi_scale/" + filename, image)

def detectMultiscale(filename):
	# load classifier
	clf = joblib.load(HOG_CLF_FILE)

	# convert to primal form
	sv_count = clf.support_vectors_.shape[0] # number of support vector
	var_count = clf.support_vectors_.shape[1] # number of features

	# get the alphas
	alphas = np.abs(clf.dual_coef_)

	# new primal support vectors
	primal_svs = np.zeros((var_count, 1))
	for r in range(sv_count):
		alpha = alphas[0][r]
		v = clf.support_vectors_[r]
		for j in range(var_count):
			primal_svs[j] += (-alpha) * v[j]

	# set up multi-scale detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(primal_svs)

	# read image
	filepath = "test/ship/multi_scale/" + filename
	image = cv2.imread(filepath, 0)

	# find ships position
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
	cv2.imwrite("results/ship/multi_scale/" + filename, image)

if __name__ == '__main__':
	if not os.path.isfile(HOG_CLF_FILE):
		trainHOG()

	# detect()
	detectScene("VNR20150816_PAN.png")