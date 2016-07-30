import cv2
import csv

import numpy as np
from numpy import linalg as LA

import math
import sys
import os

from p_motionEstBM import motionEstTSS

def getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn):
	return "{:0>4d}_{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld.png".format(bt_ID, yyyy, mm, dd, hh, mn)

def getFilePathFromTime(prefix, bt_ID, yyyy, mm, dd, hh, mn):
	return prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn)

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

""" Calculate the Histogram of Orientation
	The angle is positive number in range of [0, 180]
	Params:
		num_orient: number of bins
		magnitude_threshold: minimum motion/gradient magnitude to contribute to histogram
"""
def calcHO(velX, velY, num_orient, magnitude_threshold):
	height, width = velX.shape

	# calculate absolute magnitude of the AMV/Gradient
	magnitude = np.sqrt(velX ** 2 + velY ** 2)

	# set up histogram
	hist = np.zeros((height, width, num_orient))

	# calc cell orientation histogram
	bin_w = 180 / num_orient
	for y in range(height):
		for x in range(width):
			if magnitude[y, x] >= magnitude_threshold:
				angle = (180 / np.pi) * (np.arctan2(velY[y, x], velX[y, x]) % np.pi)

				# calc orientation using bilinear interpolation

				# major bin
				m_bin = int(angle  / bin_w - 0.5) % num_orient

				neig_offset = -1
				if angle > bin_w * (m_bin + 0.5):
					neig_offset = 1
				
				hist[y, x, m_bin] += magnitude[y, x] * abs(angle - bin_w * (m_bin + neig_offset + 0.5) ) / bin_w
				hist[y, x, m_bin + neig_offset] += magnitude[y, x] * abs(angle - bin_w * (m_bin + 0.5)) / bin_w
	return hist

""" Calculate integral image of Histogram of Oriented "Amotpheric Motion Vector"/Gradient
"""
def calcIHO(velX, velY, num_orient, magnitude_threshold):
	hist = calcHO(velX, velY, num_orient, magnitude_threshold)
	
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

""" Calculate the descriptor of HOG/HOAMV
	Params:
		ppc: pixels per cell
		cpb: cells per block
"""
def calcHODescriptor(integral_hist, ppc, cpb):
	height, width, num_orient = integral_hist.shape
	epsilon = 0.1
	tau = 0.2
	descriptor = []

	# calculate cell histogram and block-normalization
	for y in xrange(0, height - (cpb - 1) * ppc, ppc / 2):
		for x in xrange(0, width - (cpb - 1) * ppc, ppc / 2):
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
	# calculate the integral image of HOAMV/HOG
	num_orient = 9
	amv_threshold = 1
	integral_hist = calcIHO(velX, velY, num_orient, amv_threshold)

	# calculate the descriptor of HOAMV
	pixel_per_cell = 8
	cell_per_block = 2
	descriptor = calcHODescriptor(integral_hist, pixel_per_cell, cell_per_block)

	return descriptor

""" Calculate Amotpheric Motion Vector using Block Matching algorithm
"""
def calcAMVBM(im, ref_im):
	velX, velY = motionEstTSS(im, ref_im, 17, 4, 10)

	return velX, velY

""" Prepare the training images by cropping around best track point
"""
def prepareTrainImages(btTrainFile, trainFile):
	prefix = "../../data/tc/images/"
	with open(btTrainFile, 'rb') as btfile, open(trainFile, 'wb') as tFile:
		reader = csv.reader(btfile, delimiter=',')
		writer = csv.writer(tFile, delimiter=',')
		
		index = 0
		for row in reader:
			print "Processing TC:", row[7]
			bt_ID = int(row[1])
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = reader.next()
				
				status = line[2]

				datetime = line[0]
				yyyy = 2000 + int(datetime[0:2])
				mm = (int)(datetime[2:4])
				dd = (int)(datetime[4:6])
				hh = (int)(datetime[6:8])

				for mn in [00, 10]:
					impath = getFilePathFromTime(prefix, bt_ID, yyyy, mm, dd, hh, mn)
					if not os.path.isfile(impath):
						continue

					im = cv2.imread(impath, 0)
					height, width = im.shape

					bt_lat = int(line[3]) * 0.1
					bt_lon = int(line[4]) * 0.1

					row, col = latlon2xy(bt_lat, bt_lon)

					w = 167
					if row - w > 0 and row + w + 1 < height and col - 1 > 0 and col + w + 1 < width:
						crop_im = im[row - w : row + w + 1, col - w : col + w + 1]
						
						writer.writerow([getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn), status])
						cv2.imwrite("../../train/tc/pos/" + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn), crop_im)

""" Training the classifier
"""
def train(bestTrackFile):
	prefix = "../../train/tc/pos/"
	with open(bestTrackFile, 'rb') as btfile:
		reader = csv.reader(btfile, delimiter=',')		
		for row in reader:
			print "Processing TC:", row[7]
			
			bt_ID = int(row[1])
			
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = reader.next()
				
				# type of TC
				tc_type = int(line[2])
				
				# get the datetime of the image
				datetime = line[0]
				yyyy = 2000 + int(datetime[0:2])
				mm = (int)(datetime[2:4])
				dd = (int)(datetime[4:6])
				hh = (int)(datetime[6:8])

				# read image at best-track time
				impath = getFilePathFromTime(prefix, bt_ID, yyyy, mm, dd, hh, 00)
				im = cv2.imread(impath, 0)

				# read image at 10-minutes later
				ref_im_path = getFilePathFromTime(prefix, bt_ID, yyyy, mm, dd, hh, 10)
				ref_im = cv2.imread(ref_im_path, 0)

				if im.shape != ref_im.shape:
					continue
				
				# calculate the AMV
				velX, velY = calcAMVBM(im, ref_im)
				print velX.shape

				# calculate the HOG
				# descriptor = calcDescriptor(velX, velY)

if __name__ == '__main__':
	# prepare training images
	btTrainFile = "../../data/tc/besttrack/train.csv"
	trainFile = "../../train/tc/pos.csv"

	
	prepareTrainImages(btTrainFile, trainFile)
	# train(trainFile)

			