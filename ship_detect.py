
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

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
	trainPositiveFiles = getAllFilesInDirectory("train/ship/pos/*.png")
	trainNegativeFiles = getAllFilesInDirectory("train/ship/neg/*.png")

	labels = np.array([0 for i in range(len(trainNegativeFiles))] + \
						[1 for i in range(len(trainPositiveFiles))])

	list_hog_fd = []
	for filepath in (trainNegativeFiles + trainPositiveFiles):
		im = cv2.imread(filepath, 0)
		fd = hog(im, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		list_hog_fd.append(fd)
	hog_features = np.array(list_hog_fd, 'float64')

	clf = LinearSVC()
	clf.fit(hog_features, labels)
	joblib.dump(clf, HOG_CLF_FILE, compress=3)

def findShipsInImage():
	# list of test images
	test_images = []
	for filename in test_images:
		filepath = FILEPATH_PREFIX + filename + ".tif"
		vispath = FILEPATH_PREFIX + filename + ".png"
		
		print filepath
		if os.path.isfile(filepath) and os.path.isfile(vispath):
			# read color visualizable image
			vis = cv2.imread(vispath)
			greyscale = cv2.imread(vispath, 0)

			# read pan image
			dataset = gdal.Open(filepath, GA_ReadOnly)
			cols = dataset.RasterXSize
			rows = dataset.RasterYSize
			num_bands = dataset.RasterCount
			
			band = dataset.GetRasterBand(1)

			# calculate rxd image
			print "Calculating RXD"
			rxd = calculateRXD(band, rows, cols)

			# threshold the image
			print "Thresholding image"
			anomaly = thresholdAnomaly(rxd, rows, cols)

			# finding ROI positions
			print "Finding potential candidates"

			pan = band.ReadAsArray().astype(np.int)

			# noise removal
			kernel = np.ones((5, 5), np.uint8)
			opening = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, kernel, iterations = 2)

			# connected compponents
			kernel = np.ones((5, 5), np.uint8)
			im_th = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 2)

			ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			rects = [cv2.boundingRect(ctr) for ctr in ctrs]
			
			for rect in rects:
				leng = int(rect[3] * 1.6)
				pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
				pt1 = pt1 if pt1 > 0 else 0

				pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
				pt2 = pt2 if pt2 > 0 else 0

				roi = greyscale[pt1 : pt1 + leng if pt1 + leng < rows - 1 else rows - 1, \
							pt2 : pt2 + leng if pt2 + leng < cols - 1 else cols - 1]
				
				roi = cv2.resize(roi, (28, 28))
				roi = cv2.dilate(roi, (3, 3))

				fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

				# draw contour

				# save the vis
				cv2.imwrite("results/" + filename + ".png", vis)
		else:
			print "File Not Found"

def test():
	# load classifier
	clf = joblib.load(HOG_CLF_FILE)

	testFiles = getAllFilesInDirectory("test/ship/*.png")
	for filepath in testFiles:
		im = cv2.imread(filepath, 0)
		hog_fd = hog(im, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
		nbr = clf.predict(np.array([hog_fd], 'float64'))

		print filepath, nbr[0]

if __name__ == '__main__':
	# if not os.path.isfile(HOG_CLF_FILE):
	# 	trainHOG()

	# test()