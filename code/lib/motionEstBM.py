import os
import numpy as np
import cv2
from _motionEstBM import motionEstTSS

def motionEst(im, im_ref, pixels_per_cell, stepSize):
	""" Computes motion vectors using 3-step search method
		Input:
			curI: The image for which we want to find motion vectors
			nextI: The reference image
			pixels_per_cell:
		 	stepSize:
		Ouput:
		    vx, vy : the motion vectors for each direction
	"""
	# check if two images have the same size
	if im.shape != im_ref.shape:
		print "Two images do not have the same size"
		return [], []

	# get pre-defined size
	sy, sx = im.shape
	cx, cy = pixels_per_cell

	n_cellsx = int(np.floor(sx // cx))
	n_cellsy = int(np.floor(sy // cy))

	vx = np.zeros((n_cellsy, n_cellsx), dtype=np.int_)
	vy = np.zeros((n_cellsy, n_cellsx), dtype=np.int_)

	motionEstTSS(im, im_ref, sx, sy, n_cellsx, n_cellsy, stepSize, cx, cy, vx, vy)

	return vx, vy

def test():
	im = cv2.imread("../../data/tc/full/1509_201507070000.tir.01.fld.png", 0)
	im_ref = cv2.imread("../../data/tc/full/1509_201507070010.tir.01.fld.png", 0)
	print im.shape, im_ref.shape

	vx, vy = motionEst(im, im_ref, (16, 16), 8)

	for i in range(0, len(vy)):
		for j in range(0, len(vx)):
			anchorY = int((i + 0.5) * 16)
			anchorX = int((j + 0.5) * 16)

			cv2.circle(im, (anchorY, anchorX), 1, (0, 0, 255), 1)
			cv2.line(im, (anchorY, anchorX), (anchorY + int(vy[i, j]), anchorX + int(vx[i, j])), (0, 255, 0), 1, cv2.CV_AA)
	cv2.imwrite("motion.png", im)

if __name__ == '__main__':
	test()