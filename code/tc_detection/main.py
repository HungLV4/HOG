import cv2
import csv

import numpy as np

from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from skimage.feature import hog

import math
import sys
import os

import shutil

from pandas import DataFrame

from p_motionEstBM import motionEstTSS
from descriptor import *
from utilities import *

im_full_prefix = "../../data/tc/full/"
im_area_prefix = "../../data/tc/area/"

train_full_bt_filepath = "../../data/tc/besttrack/train.csv"
test_full_bt_filepath = "../../data/tc/besttrack/test.csv"

train_area_bt_filepath = "../../train/tc/data.csv"
test_area_bt_filepath = "../../test/tc/data.csv"

amv_prefix = "../../genfiles/amv/"
amv_vis_prefix = "../../genfiles/amv_visualization/"

gradient_prefix = "../../genfiles/gradient/"

CLF_FILE = "../../genfiles/tc_clf.pkl"

""" Calculate the descripptor
"""
def calcDescriptor(velX, velY):
	# crop to smaller region of 32x32
	# height, width = velX.shape
	# row_lower_bound = height / 2 - 1
	# row_upper_bound = height / 2 + 1
	# col_lower_bound = width / 2 - 1
	# col_upper_bound = width / 2 + 1

	# velX = velX[row_lower_bound - 15: row_upper_bound + 15, col_lower_bound - 15: col_upper_bound + 15]
	# velY = velY[row_lower_bound - 15: row_upper_bound + 15, col_lower_bound - 15: col_upper_bound + 15]

	# calculate the descriptor of HOAMV
	num_orient = 9
	magnitude_threshold = 0.001

	pixel_per_cell = 16
	cell_per_block = 2
	descriptor = calcHODescriptor(velX, velY, num_orient, magnitude_threshold, pixel_per_cell, cell_per_block)
	
	# descriptor = calcSSHDescriptor(velX, velY, num_orient, amv_threshold)

	return descriptor

""" Calculate the Amospheric Motion Vector
"""
def calcAMVImages(bt_filepath):
	with open(bt_filepath, 'rb') as file:
		reader = csv.reader(file, delimiter=',')		
		for line in reader:		
			bt_ID = int(line[0])
			tc_type = int(line[2])
			
			# get the datetime of the image
			datetime = line[1]
			yyyy = 2000 + int(datetime[0:2])
			mm = (int)(datetime[2:4])
			dd = (int)(datetime[4:6])
			hh = (int)(datetime[6:8])

			# read image at best-track time
			impath = getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 00)
			im = cv2.imread(impath, 0)

			# read image at 10-minutes later
			ref_im_path = getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 10)
			ref_im = cv2.imread(ref_im_path, 0)

			print "Processing:", impath, ref_im_path
			
			if im.shape != ref_im.shape:
				continue

			# calculate the AMV
			velX, velY = motionEstTSS(im, ref_im, 25, 8, 5)

			# save the AMV for later use
			np.save(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy", velX)
			np.save(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy", velY)

""" Calculate the Gradient
"""
def calcGradientImages(bt_filepath):
	with open(bt_filepath, 'rb') as file:
		reader = csv.reader(file, delimiter=',')		
		for line in reader:		
			bt_ID = int(line[0])
			tc_type = int(line[2])
			
			# get the datetime of the image
			datetime = line[1]
			yyyy = 2000 + int(datetime[0:2])
			mm = (int)(datetime[2:4])
			dd = (int)(datetime[4:6])
			hh = (int)(datetime[6:8])

			# read image at best-track time
			impath = getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 00)
			im = cv2.imread(impath, 0)

			print "Processing:", impath
			
			# calculate the gradient
			gx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
			gy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

			# save the AMV for later use
			np.save(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy", gx)
			np.save(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy", gy)

""" Visualize the AMV
"""
def visualizeAMV(bt_filepath):
	with open(bt_filepath, 'rb') as bt_file:
		reader = csv.reader(bt_file, delimiter=',')
		for line in reader:
			bt_ID = int(line[0])
			tc_type = int(line[2])
			
			# get the datetime of the image
			datetime = line[1]
			yyyy = 2000 + int(datetime[0:2])
			mm = (int)(datetime[2:4])
			dd = (int)(datetime[4:6])
			hh = (int)(datetime[6:8])

			print bt_ID, datetime

			# load motion vector
			velX = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			velY = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")
			velSize = velX.shape

			# load image
			im_c = cv2.imread(getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 00), 1)
			for i in range(0, velSize[0]):
				for j in range(0, velSize[1]):
					anchorX = i * 5 + 12
					anchorY = j * 5 + 12

					cv2.circle(im_c, (anchorY, anchorX), 1, (0, 0, 255), 1)
					cv2.line(im_c, (anchorY, anchorX), (anchorY + velY[i, j], anchorX + velX[i, j]), (0, 255, 0), 1, cv2.CV_AA)
			cv2.imwrite(getFilePathFromTime(amv_vis_prefix, bt_ID, yyyy, mm, dd, hh, 00), im_c)

""" Training the classifier
"""
def train():
	features = []
	labels = []
	with open(train_area_bt_filepath, 'rb') as bt_file:
		reader = csv.reader(bt_file, delimiter=',')
		for line in reader:
			bt_ID = int(line[0])
			
			tc_type = int(line[2]) - 1
			labels.append(tc_type)
			
			# get the datetime of the image
			datetime = line[1]
			yyyy = 2000 + int(datetime[0:2])
			mm = (int)(datetime[2:4])
			dd = (int)(datetime[4:6])
			hh = (int)(datetime[6:8])

			# calculate the Histogram descriptor using AMV
			# velX = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			# velY = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")
			# descriptor = calcDescriptor(velX, velY)

			# calculate the HOG descriptpor using Scikit lib
			# im = cv2.imread(getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 00), 0)
			# descriptor = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)

			# calculate the HOG descriptpor
			gx = np.load(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			gy = np.load(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")
			descriptor = calcDescriptor(gx, gy)
			
			features.append(descriptor)

	clf = LinearSVC()
	clf.fit(features, labels)
	joblib.dump(clf, CLF_FILE, compress=3)

def test():
	# load clf
	clf = joblib.load(CLF_FILE)
	
	# count the correc/not-correct predict for each TC type
	confusion = np.zeros((9, 9))
	with open(test_area_bt_filepath, 'rb') as bt_file:
		reader = csv.reader(bt_file, delimiter=',')
		for line in reader:
			bt_ID = int(line[0])
			tc_type = int(line[2]) - 1
			
			# get the datetime of the image
			datetime = line[1]
			yyyy = 2000 + int(datetime[0:2])
			mm = (int)(datetime[2:4])
			dd = (int)(datetime[4:6])
			hh = (int)(datetime[6:8])

			# calculate the Histogram descriptor using AMV
			# velX = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			# velY = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")
			# descriptor = calcDescriptor(velX, velY)

			# calculate the HOG descriptpor using Scikit lib
			# im = cv2.imread(getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 00), 0)
			# descriptor = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)

			# calculate the HOG descriptpor
			gx = np.load(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			gy = np.load(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")
			descriptor = calcDescriptor(gx, gy)

			nbr = clf.predict(np.array([descriptor]))
			confusion[tc_type, int(nbr[0])] += 1
	
	# pretty-print confusion matrix
	print DataFrame(confusion)

if __name__ == '__main__':
	# cropImagesByBestTrack(im_full_prefix, im_area_prefix, train_full_bt_filepath, train_area_bt_filepath)
	# cropImagesByBestTrack(im_full_prefix, im_area_prefix, test_full_bt_filepath, test_area_bt_filepath)

	# calcAMVImages(train_area_bt_filepath)
	# visualizeAMV(train_area_bt_filepath)
	# calcAMVImages(test_area_bt_filepath)
	# visualizeAMV(test_area_bt_filepath)

	# calcGradientImages(train_area_bt_filepath)
	# calcGradientImages(test_area_bt_filepath)

	# train()
	# test()

			