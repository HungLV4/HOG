import cv2
import csv
import numpy as np
import math

def getFilePathFromTime(yyyy, mm, dd, hh, mn):
	return "../../data/tc/images/cyclic/{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld.tiff".format(yyyy, mm, dd, hh, mn)

def calcAMVBlockMatching(im, ref_im):
	height, width = im.shape

	# for visualization
	im_c = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

	# initialize block matching parameters
	blockSize = (24, 24)
	shiftSize = (10, 10)
	maxRange = (10, 10)

	velSize = ((height - blockSize[0] + shiftSize[0]) / shiftSize[0], (width - blockSize[1] + shiftSize[1]) / shiftSize[1])

	velX = cv2.cv.fromarray(np.zeros(velSize, dtype=np.float32))
	velY = cv2.cv.fromarray(np.zeros(velSize, dtype=np.float32))

	# calculate motion using block matching
	print "Calculating wind vector ..."
	cv2.cv.CalcOpticalFlowBM(cv2.cv.fromarray(im), cv2.cv.fromarray(ref_im), blockSize, shiftSize, maxRange, 0, velX, velY)

	# visualizing
	print "Visualizing ..."
	for i in range(0, velSize[0]):
		for j in range(0, velSize[1]):
			motionX = int(math.ceil(velX[i, j]))
			motionY = int(math.ceil(velY[i, j]))

			anchorX = i * shiftSize[0] + (blockSize[0] / 2)
			anchorY = j * shiftSize[1] + (blockSize[1] / 2)

			cv2.circle(im_c, (anchorY, anchorX), 1, (0, 0, 255), -1)
			cv2.line(im_c, (anchorY, anchorX), (anchorY + 2 * motionY, anchorX + 2 * motionX), (0, 255, 0), 1, cv2.CV_AA)

	cv2.imwrite("motion.png", im_c)

def calcAMVAtTime(yyyy, mm, dd, hh, mn):
	read_mode = -1

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
	if im.shape != ref_im.shape:
		return	
	
	# calculate AMV using block mathing algorithm and visualize it
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
	mm = 07
	dd = 07
	hh = 00
	mn = 00

	calcAMVAtTime(yyyy, mm, dd, hh, mn)

if __name__ == '__main__':
	# bestTrackFile = "../../data/tc/besttrack/besttrack.csv"
	# processByBestTrack(bestTrackFile)

	# testing
	test()


			