
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

import matplotlib.pyplot as plt

HOG_CLF_FILE = "genfiles/ships_hog_clf.pkl"
TRAIN_LABEL_FILE = "genfiles/ships_label.csv"
TRAIN_FEATURES_FILE = "genfiles/ships_feature.dat"

FILEPATH_PREFIX = "../../../../temp/CT_OCEAN/"

def getAllFilesInDirectory(dir):
	return glob.glob(dir)

def calcAbnormality(band):
	pass

def gradient():
	files = getAllFilesInDirectory("../../train/ship/neg/64x128/*.png")
	for filepath in files:
		im = cv2.imread(filepath, 0)

		# calculate the gradient
		gx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=5)
		gy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=5)

		# 
		orientations = np.arctan2(gy, gx) * (180 / np.pi) % 180
		magnitude = np.hypot(gy, gx)

		# hist, edges = np.histogram(orientations[1:-1, 1:-1], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])

def trainHOG():
	trainPositiveFiles = getAllFilesInDirectory("../../train/ship/pos/64x128/*.png")
	trainNegativeFiles = getAllFilesInDirectory("../../train/ship/neg/64x128/*.png")

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

def detectMultiscale(filename, winStride = (4, 4)):
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

if __name__ == '__main__':
	gradient()
	# trainHOG()
	# detect()
	# detectMultiscale("0000093.png")