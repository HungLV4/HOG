import cv2
import csv

import numpy as np

from sklearn.externals import joblib
from sklearn.svm import LinearSVC

import math
import sys
import os

import shutil

from pandas import DataFrame

from motionEstBM import motionEstTSS
from hog import calcHOGDescriptor
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

def calcDescriptor(gx, gy):
	""" Calculate the descripptor
	"""

	# calculate the descriptor of HOG/HOAMV
	descriptor = calcHOGDescriptor(gx, gy, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2))

	return descriptor

def calcAMVImages(bt_filepath):
	""" Calculate the Amospheric Motion Vector
	"""
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
			impath = getFilePathFromTime(im_area_prefix, tc_type, bt_ID, yyyy, mm, dd, hh, 00)
			im = cv2.imread(impath, 0)

			# read image at 10-minutes later
			ref_im_path = getFilePathFromTime(im_area_prefix, tc_type, bt_ID, yyyy, mm, dd, hh, 10)
			ref_im = cv2.imread(ref_im_path, 0)

			print "Processing:", impath
			
			if im.shape != ref_im.shape:
				continue

			# calculate the AMV
			velX, velY = motionEstTSS(im, ref_im, 17, 8, 8)

			# save the AMV for later use
			np.save(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy", velX)
			np.save(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy", velY)

def calcGradientImages(bt_filepath):
	""" Calculate the Gradient
	"""
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
			impath = getFilePathFromTime(im_area_prefix, tc_type, bt_ID, yyyy, mm, dd, hh, 00)
			im = cv2.imread(impath, 0)
			im = im.astype('float')

			print "Processing:", impath
			
			# calculate the gradient
			gx = np.empty(im.shape, dtype=np.double)
			gx[:, 0] = 0
			gx[:, -1] = 0
			gx[:, 1:-1] = im[:, 2:] - im[:, :-2]

			gy = np.empty(im.shape, dtype=np.double)
			gy[0, :] = 0
			gy[-1, :] = 0
			gy[1:-1, :] = im[2:, :] - im[:-2, :]

			# save the AMV for later use
			np.save(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy", gx)
			np.save(gradient_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy", gy)

def visualizeAMV(bt_filepath):
	""" Visualize the AMV
	"""
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
			im_c = cv2.imread(getFilePathFromTime(im_area_prefix, tc_type, bt_ID, yyyy, mm, dd, hh, 00), 1)
			for i in range(0, velSize[0]):
				for j in range(0, velSize[1]):
					anchorX = i * 5 + 12
					anchorY = j * 5 + 12

					cv2.circle(im_c, (anchorY, anchorX), 1, (0, 0, 255), 1)
					cv2.line(im_c, (anchorY, anchorX), (anchorY + velY[i, j], anchorX + velX[i, j]), (0, 255, 0), 1, cv2.CV_AA)
			cv2.imwrite(getFilePathFromTime(amv_vis_prefix, tc_type, bt_ID, yyyy, mm, dd, hh, 00), im_c)

def train(data_prefix):
	""" Training the classifier
	"""
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
			gx = np.load(data_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			gy = np.load(data_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")

			gx = gx.astype(np.double) 
			gy = gy.astype(np.double) 
			descriptor = calcDescriptor(gx, gy)
		
			features.append(descriptor)

	clf = LinearSVC()
	clf.fit(features, labels)
	joblib.dump(clf, CLF_FILE, compress=3)

def test(data_prefix):
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
			gx = np.load(data_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			gy = np.load(data_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")

			gx = gx.astype(np.double) 
			gy = gy.astype(np.double) 
			descriptor = calcDescriptor(gx, gy)

			nbr = clf.predict(np.array([descriptor]))
			confusion[tc_type, int(nbr[0])] += 1
	
	# pretty-print confusion matrix
	print DataFrame(confusion)

if __name__ == '__main__':
	# cropImagesByBestTrack(im_full_prefix, im_area_prefix, train_full_bt_filepath, train_area_bt_filepath)
	# cropImagesByBestTrack(im_full_prefix, im_area_prefix, test_full_bt_filepath, test_area_bt_filepath)

	m1 = int(sys.argv[1])
	m2 = int(sys.argv[2])
	
	if m1 == 1:
		calcGradientImages(train_area_bt_filepath)
		calcGradientImages(test_area_bt_filepath)
	elif m1 == 2:
		calcAMVImages(train_area_bt_filepath)
		visualizeAMV(train_area_bt_filepath)
		
		calcAMVImages(test_area_bt_filepath)
		visualizeAMV(test_area_bt_filepath)

	if m2 == 1:
		train(gradient_prefix)
		test(gradient_prefix)
	elif m2 == 2:
		train(amv_prefix)
		test(amv_prefix)

			