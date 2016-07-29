import cv2
import csv
import numpy as np
import math
import sys

from p_motionEstBM import motionEstTSS

def getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn):
	return "{:0>4d}_{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld.png".format(bt_ID, yyyy, mm, dd, hh, mn)

def getFilePathFromTime(prefix, bt_ID, yyyy, mm, dd, hh, mn):
	return prefix + getFileNameFromTime(bt_ID, yyyy, mm, dd, hh, mn)

def latlon2xy(lat, lon):
	if lat > 60 or lat < -60 or lon > 205 or lon < 85:
		return -1, -1

	return int((60 - lat) / 0.067682), int((lon - 85) / 0.067682)

""" Calculate integral image of Histogram of Oriented Amotpheric Motion Vector
"""
def calcIHOAMV(im, ref_im, num_orient, magnitude_threshold):
	# get the motion vector
	velX, velY = calcAMVBM(im, ref_im)

	# calculate the Histogram of Oriented Amotpheric Motion Vector
	height, width = velX.shape

	# calculate absolute magnitude of the AMV
	magnitude = np.sqrt(velX ** 2 + velY ** 2)

	# set up histogram
	hist = np.zeros((height, width, num_orient))

	# calc initial orientation histogram
	mid = num_orient / 2
	for y in range(height):
		for x in range(width):
			if magnitude[y, x] >= magnitude_threshold:
				angle = np.arctan2(velY[y, x], velX[y, x])
				orientation = int(math.floor(1 + angle / (np.pi / mid)))
				
				hist[y, x, orientation] += magnitude[y, x]

	# calc integral image of Histogram of Oriented Amotpheric Motion Vector
	integral_hist = np.copy(hist)
	for y in xrange(1, height):
		for x in xrange(1, width):
			for ang in range(num_orient):
				integral_hist[y, x, ang] += hist[y, x, ang] + \
											hist[y - 1, x, ang] + \
											hist[y, x - 1, ang] - \
											hist[y - 1, x - 1, ang]

	return integral_hist

""" Calculate Amotpheric Motion Vector using Block Matching algorithm
"""
def calcAMVBM(im, ref_im):
	velX, velY = motionEstTSS(im, ref_im, 16, 4, 10)

	return velX, velY

""" Prepare the training images by cropping around best track point
"""
def prepareTrainImages(bestTrackFile):
	prefix = "../../data/tc/images/"
	with open(bestTrackFile, 'rb') as btfile:
		reader = csv.reader(btfile, delimiter=',')
		
		index = 0
		for row in reader:
			print "Processing TC:", row[7]
			bt_ID = int(row[1])
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = reader.next()
				datetime = line[0]
				
				yyyy = 2000 + int(datetime[0:2])
				mm = (int)(datetime[2:4])
				dd = (int)(datetime[4:6])
				hh = (int)(datetime[6:8])

				for mn in [00, 10]:
					impath = getFilePathFromTime(prefix, bt_ID, yyyy, mm, dd, hh, mn)
					print impath

					im = cv2.imread(impath, 0)
					height, width = im.shape

					bt_lat = int(line[3]) * 0.1
					bt_lon = int(line[4]) * 0.1

					row, col = latlon2xy(bt_lat, bt_lon)

					w = 200
					crop_im = im[row - w if row - w > 0 else 0: row + w if row + w < height else height - 1, 
								col - w if col - w > 0 else 0 : col + w if col + w < width else width - 1]

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
				
				# calculate the integral image of HOAMV
				integral_hist = calcIHOAMV(im, ref_im, 9, 1)

				# calculate the descriptor of HOAMV

if __name__ == '__main__':
	# prepare training images
	trainBTFile = "../../data/tc/besttrack/train.csv"
	
	# prepareTrainImages(trainBTFile)
	train(trainBTFile)

			