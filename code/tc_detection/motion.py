import cv2
import csv
import numpy as np
import math

from motionEstBM import motionEstTSS

def getFilePathFromTime(yyyy, mm, dd, hh, mn):
	return "../../data/tc/images/gray/{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld.tiff".format(yyyy, mm, dd, hh, mn)

def calcAMVBlockMatching(im, ref_im):
	# for visualization
	im_c = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	
	# calculate AMV using block mathing algorithm and visualize it
	velX, velY = motionEstTSS(im, ref_im, 8, 4, 10)

	if len(velX) == 0 or len(velY) == 0:
		return
	
	# visualization
	velSize = velX.shape
	for i in range(0, velSize[0]):
		for j in range(0, velSize[1]):
			anchorX = i * 10
			anchorY = j * 10

			cv2.circle(im_c, (anchorY, anchorX), 1, (0, 0, 255), 1)
			cv2.line(im_c, (anchorY, anchorX), (anchorY + 2 * velY[i, j], anchorX + 2 * velX[i, j]), (0, 255, 0), 1, cv2.CV_AA)
	cv2.imwrite("motion.png", im_c)

def calcAMVAtTime(yyyy, mm, dd, hh, mn):
	read_mode = 0

	# read image at best-track time
	impath = getFilePathFromTime(yyyy, mm, dd, hh, mn)
	im = cv2.imread(impath, read_mode)
	if read_mode == -1:
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	# read image at 10-minutes later
	ref_im_path = getFilePathFromTime(yyyy, mm, dd, hh, mn + 10)
	ref_im = cv2.imread(ref_im_path, read_mode)
	if read_mode == -1:
		ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)

	# check if two images have the same size	
	calcAMVBlockMatching(im, ref_im)

def processByBestTrack(bestTrackFile):
	with open(bestTrackFile, 'rb') as btfile:
		reader = csv.reader(btfile, delimiter=',')
		
		index = 0
		for row in reader:
			print "Processing TC:", row[7]
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = reader.next()
				datetime = line[0]
				
				yyyy = 2000 + int(datetime[0:2])
				mm = (int)(datetime[2:4])
				dd = (int)(datetime[4:6])
				hh = (int)(datetime[6:8])
				mn = 00

				calcAMVAtTime(yyyy, mm, dd, hh, mn)

def test():
	yyyy = 2015
	mm = 7
	dd = 7
	hh = 0
	mn = 0

	calcAMVAtTime(yyyy, mm, dd, hh, mn)

if __name__ == '__main__':
	# bestTrackFile = "../../data/tc/besttrack/besttrack.csv"
	# processByBestTrack(bestTrackFile)

	# testing
	test()


			