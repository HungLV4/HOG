import cv2
import csv

import numpy as np
from numpy import linalg as LA

from sklearn.externals import joblib
from sklearn.svm import LinearSVC

import math
import sys
import os

import shutil

from pandas import DataFrame

from p_motionEstBM import motionEstTSS

CLF_FILE = "../../genfiles/tc_clf.pkl"

def getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn):
	return "{:0>4d}_{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld".format(bt_ID, yyyy, mm, dd, hh, mn)

def getFilePathFromTime(prefix, bt_ID, yyyy, mm, dd, hh, mn):
	return prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn) + ".png"

def latlon2xy(lat, lon):
	if lat > 60 or lat < -60 or lon > 205 or lon < 85:
		return -1, -1

	return int((60 - lat) / 0.067682), int((lon - 85) / 0.067682)

""" L2-Normalization for a list
	Params:
		x: list of value
"""
def l2_normalization(x):
	l2_norm  = LA.norm(x)
	if l2_norm > 0:
		for i in range(len(x)):
			x[i] = x[i] / l2_norm
	return x

""" Calculate the Histogram of Wind Speed
"""
def calcHWS(velX, velY, num_bin):
	height, width = velX.shape

	# calculate absolute magnitude of the AMV/Gradient
	magnitude = np.sqrt(velX ** 2 + velY ** 2)

	hist = np.zeros(num_bin + 1)
	for y in range(height):
		for x in range(width):
			b = int(magnitude[y, x]) if int(magnitude[y, x]) < num_bin else num_bin

			hist[b] += 1
	return hist

""" Calculate the Histogram of Wind Direction
	The angle is positive number in range of [0, 180]
	Params:
		num_orient: number of bins
		magnitude_threshold: minimum motion/gradient magnitude to contribute to histogram
"""
def caclcHWD(velX, velY, num_orient, amv_threshold):
	height, width = velX.shape

	# calculate absolute magnitude of the AMV/Gradient
	magnitude = np.sqrt(velX ** 2 + velY ** 2)

	# set up histogram
	hist = np.zeros((height, width, num_orient))

	# calc cell orientation histogram
	bin_w = 180 / num_orient
	for y in range(height):
		for x in range(width):
			if magnitude[y, x] >= amv_threshold:
				angle = (180 / np.pi) * (np.arctan2(velY[y, x], velX[y, x]) % np.pi)

				# calc orientation using bilinear interpolation

				# major bin
				m_bin = int(angle  / bin_w)
				if m_bin >= num_orient:
					m_bin = num_orient - 1

				neig_offset = -1
				if angle > bin_w * (m_bin + 0.5):
					neig_offset = 1
				
				if m_bin == 0 and neig_offset == -1:
					hist[y, x, 8] = magnitude[y, x] * abs(angle - bin_w * (0 + 0.5)) / bin_w
					angle = 180 + angle
					hist[y, x, 0] = magnitude[y, x] * abs(angle - bin_w * (8 + 0.5)) / bin_w
				elif m_bin == 8 and neig_offset == 1:
					hist[y, x, 0] += magnitude[y, x] * abs(angle - bin_w * (8 + 0.5)) / bin_w
					angle = angle - 180
					hist[y, x, 8] += magnitude[y, x] * abs(angle - bin_w * (0 + 0.5)) / bin_w
				else:
					hist[y, x, m_bin] += magnitude[y, x] * abs(angle - bin_w * (m_bin + neig_offset + 0.5) ) / bin_w
					hist[y, x, m_bin + neig_offset] += magnitude[y, x] * abs(angle - bin_w * (m_bin + 0.5)) / bin_w

	return hist

""" Calculate integral image of Histogram of Oriented "Amotpheric Motion Vector"/Gradient
"""
def calcIHO(velX, velY, num_orient, magnitude_threshold):
	hist = caclcHWD(velX, velY, num_orient, magnitude_threshold)
	
	height, width, _ = hist.shape

	# calc integral image of HOAMV/HOG
	integral_hist = np.copy(hist)
	for y in xrange(1, height):
		for x in xrange(1, width):
			for ang in range(num_orient):
				integral_hist[y, x, ang] += hist[y, x, ang] + \
											hist[y - 1, x, ang] + \
											hist[y, x - 1, ang] - \
											hist[y - 1, x - 1, ang]

	return integral_hist

""" Calculate the descriptor of Shen-Shyang Ho paper
	*Automated cyclone identification from remote QuickSCAT satellite data
"""
def calcSSHDescriptor(velX, velY, num_orient, amv_threshold):
	hist_wd = caclcHWD(velX, velY, num_orient, amv_threshold)
	wd_des = np.sum(hist_wd, (0, 1))

	ws_des = calcHWS(velX, velY, 22)

	return  np.concatenate([wd_des, ws_des])

""" Calculate the descriptor of Circulate Histogram of Oriented AMV/Gradient
"""
def calcCHODescriptor(velX, velY, num_orient, amv_threshold, radius):
	hist = caclcHWD(velX, velY, num_orient, amv_threshold)

	height, width = velX.shape
	
	row_lower_bound = height / 2 - 1
	row_upper_bound = height / 2 + 1
	col_lower_bound = width / 2 - 1
	col_upper_bound = width / 2 + 1

""" Calculate the descriptor of HOG/HOAMV
	Params:
		ppc: pixels per cell
		cpb: cells per block
"""
def calcHODescriptor(velX, velY, num_orient, amv_threshold, ppc, cpb):
	integral_hist = calcIHO(velX, velY, num_orient, amv_threshold)

	height, width, num_orient = integral_hist.shape
	epsilon = 0.1
	tau = 0.2
	descriptor = []

	# calculate cell histogram and block-normalization
	for y in xrange(0, height - (cpb - 1) * ppc, ppc):
		for x in xrange(0, width - (cpb - 1) * ppc, ppc):
			# block feature
			block_features = []
			
			if y + (cpb - 1) * ppc + (ppc - 1) < height and x + (cpb - 1) * ppc + (ppc - 1) < width:	
				# calculate hog vector for each cell in block
				for i in range(cpb):
					for j in range(cpb):				
						for b in range(num_orient):
							val = integral_hist[y + i * ppc, x + j * ppc, b] + \
									integral_hist[y + i * ppc + (ppc - 1), x + j * ppc + (ppc - 1), b] - \
									integral_hist[y + i * ppc + (ppc - 1), x + j * ppc, b] - \
									integral_hist[y + i * ppc, x + j * ppc + (ppc - 1), b]
							block_features.append(val)
			# L2-norm block normalization
			block_features = l2_normalization(block_features)
			descriptor = descriptor + block_features
	
	# change to numpy array for computation
	descriptor = np.array(descriptor)
	
	# L2-norm
	descriptor = l2_normalization(descriptor)
	
	# filter out large value
	descriptor[descriptor > tau] = tau

	# L2-norm again
	descriptor = l2_normalization(descriptor)

	return descriptor

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
	amv_threshold = 1
	
	# pixel_per_cell = 8
	# cell_per_block = 2
	# descriptor = calcHODescriptor(velX, velY, num_orient, amv_threshold, pixel_per_cell, cell_per_block)
	
	descriptor = calcSSHDescriptor(velX, velY, num_orient, amv_threshold)

	return descriptor

""" Calculate Amotpheric Motion Vector using Block Matching algorithm
"""
def calcAMVBM(im, ref_im):
	velX, velY = motionEstTSS(im, ref_im, 25, 8, 5)

	return velX, velY

""" Prepare the training images by cropping around best track point
"""
def cropImagesByBestTrack(im_full_prefix, im_area_prefix, full_bt_filepath, area_bt_filepath):
	with open(full_bt_filepath, 'rb') as full_file, open(area_bt_filepath, 'wb') as area_file:
		reader = csv.reader(full_file, delimiter=',')
		writer = csv.writer(area_file, delimiter=',')
		
		index = 0
		for row in reader:
			print "Processing TC:", row[7]
			bt_ID = int(row[1])
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = reader.next()
				
				tc_type = line[2]

				datetime = line[0]

				yyyy = 2000 + int(datetime[0:2])
				mm = (int)(datetime[2:4])
				dd = (int)(datetime[4:6])
				hh = (int)(datetime[6:8])

				impath = getFilePathFromTime(im_full_prefix, bt_ID, yyyy, mm, dd, hh, 00)
				ref_impath = getFilePathFromTime(im_full_prefix, bt_ID, yyyy, mm, dd, hh, 10)
				
				if not (os.path.isfile(impath) and os.path.isfile(ref_impath)):
					continue

				im = cv2.imread(impath, 0)
				ref_im = cv2.imread(ref_impath, 0)

				height, width = im.shape

				bt_lat = int(line[3]) * 0.1
				bt_lon = int(line[4]) * 0.1

				row, col = latlon2xy(bt_lat, bt_lon)

				w = 172
				if row - w > 0 and row + w + 1 < height and col - 1 > 0 and col + w + 1 < width:
					crop_im = im[row - w : row + w + 1, col - w : col + w + 1]
					crop_ref = ref_im[row - w : row + w + 1, col - w : col + w + 1]
					
					cv2.imwrite(getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 00), crop_im)
					cv2.imwrite(getFilePathFromTime(im_area_prefix, bt_ID, yyyy, mm, dd, hh, 10), crop_ref)

					writer.writerow([bt_ID, datetime, tc_type])

""" Training the classifier
"""
def calcAMVImages(im_prefix, amv_prefix, bt_filepath):
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
			impath = getFilePathFromTime(im_prefix, bt_ID, yyyy, mm, dd, hh, 00)
			im = cv2.imread(impath, 0)

			# read image at 10-minutes later
			ref_im_path = getFilePathFromTime(im_prefix, bt_ID, yyyy, mm, dd, hh, 10)
			ref_im = cv2.imread(ref_im_path, 0)

			print "Processing:", impath, ref_im_path
			
			if im.shape != ref_im.shape:
				continue

			# calculate the AMV
			velX, velY = calcAMVBM(im, ref_im)

			# save the AMV for later use
			np.save(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy", velX)
			np.save(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy", velY)

def visualizeAMV(amv_prefix, amv_vis_prefix, im_prefix, bt_filepath):
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
			print velSize

			# load image
			im_c = cv2.imread(getFilePathFromTime(im_prefix, bt_ID, yyyy, mm, dd, hh, 00), 1)
			for i in range(0, velSize[0]):
				for j in range(0, velSize[1]):
					anchorX = i * 5 + 12
					anchorY = j * 5 + 12

					cv2.circle(im_c, (anchorY, anchorX), 1, (0, 0, 255), 1)
					cv2.line(im_c, (anchorY, anchorX), (anchorY + velY[i, j], anchorX + velX[i, j]), (0, 255, 0), 1, cv2.CV_AA)
			cv2.imwrite(getFilePathFromTime(amv_vis_prefix, bt_ID, yyyy, mm, dd, hh, 00), im_c)

def train(bt_filepath, amv_prefix):
	features = []
	labels = []
	with open(bt_filepath, 'rb') as bt_file:
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

			velX = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			velY = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")

			# calculate the HOG
			descriptor = calcDescriptor(velX, velY)
			features.append(descriptor)
			labels.append(tc_type)
	
	clf = LinearSVC()
	clf.fit(features, labels)
	joblib.dump(clf, CLF_FILE, compress=3)

def test(test_area_bt_filepath, amv_prefix):
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

			velX = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_X.npy")
			velY = np.load(amv_prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, 00) + "_Y.npy")

			descriptor = calcDescriptor(velX, velY)
			nbr = clf.predict(np.array([descriptor]))

			confusion[tc_type, int(nbr[0])] += 1
	
	# pretty-print confusion matrix
	print DataFrame(confusion)

if __name__ == '__main__':
	im_full_prefix = "../../data/tc/full/"
	im_area_prefix = "../../data/tc/area/"
	
	train_full_bt_filepath = "../../data/tc/besttrack/train.csv"
	test_full_bt_filepath = "../../data/tc/besttrack/test.csv"
	
	train_area_bt_filepath = "../../train/tc/data.csv"
	test_area_bt_filepath = "../../test/tc/data.csv"

	amv_prefix = "../../genfiles/amv/"
	amv_vis_prefix = "../../genfiles/amv_visualization/"

	# cropImagesByBestTrack(im_full_prefix, im_area_prefix, train_full_bt_filepath, train_area_bt_filepath)
	# cropImagesByBestTrack(im_full_prefix, im_area_prefix, test_full_bt_filepath, test_area_bt_filepath)

	# calcAMVImages(im_area_prefix, amv_prefix, train_area_bt_filepath)
	# calcAMVImages(im_area_prefix, amv_prefix, test_area_bt_filepath)
	
	# visualizeAMV(amv_prefix, amv_vis_prefix, im_area_prefix, train_area_bt_filepath)
	# visualizeAMV(amv_prefix, amv_vis_prefix, im_area_prefix, test_area_bt_filepath)

	train(train_area_bt_filepath, amv_prefix)
	test(test_area_bt_filepath, amv_prefix)

			