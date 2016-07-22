import cv2
import csv
import numpy as np
import math

def getFilePathFromTime(yyyy, mm, dd, hh, mn):
	return "../../data/tc/images/rainbow/{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld.tiff".format(yyyy, mm, dd, hh, mn)

def calcWindAtTime(yyyy, mm, dd, hh, mn):
	# read image at best-track time
	impath = getFilePathFromTime(yyyy, mm, dd, hh, mn)
	im = cv2.imread(impath, -1)

	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	
	# read image at 10-minutes later
	ref_im_path = getFilePathFromTime(yyyy, mm, dd, hh, mn + 10)
	ref_im = cv2.imread(impath, -1)

	ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)

	height, width = im.shape

	# calculate dense optical flow
	# flow = cv2.calcOpticalFlowFarneback(im, ref_im, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

	# calculate motion using block matching
	blockSize  = (4, 4)
	shiftSize = (1, 1)
	maxRange = (10, 10)
	velX = np.zeros((height, width), dtype=np.float32)
	velY = np.zeros((height, width), dtype=np.float32)

	cv2.cv.CalcOpticalFlowBM(cv2.cv.fromarray(im), cv2.cv.fromarray(ref_im), blockSize, shiftSize, maxRange, 0, cv2.cv.fromarray(velX), cv2.cv.fromarray(velY))

	# for i in xrange(0, height, 20):
	# 	for j in xrange(0, width, 20):
	# 		motion = flow[i, j]
	# 		cv2.line(im, (i, j), (i + 10 * int(math.ceil(motion[0])), j + 10 * int(math.ceil(motion[1]))), (0, 0, 0), 2, cv2.CV_AA)

	# cv2.imwrite("motion.png", im)

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

				calcWindAtTime(yyyy, mm, dd, hh, mn)

def test():
	yyyy = 2015
	mm = 07
	dd = 07
	hh = 00
	mn = 00

	calcWindAtTime(yyyy, mm, dd, hh, mn)

if __name__ == '__main__':
	# bestTrackFile = "../../data/tc/besttrack/besttrack.csv"
	# processByBestTrack(bestTrackFile)

	# testing
	test()


			