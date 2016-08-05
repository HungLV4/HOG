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

	vx = np.zeros((n_cellsy, n_cellsx))
	vy = np.zeros((n_cellsy, n_cellsx))

	motionEstTSS(im, im_ref, sx, sy, cx, cy, stepSize, n_cellsx, n_cellsy, vx, vy)

	return vx, vy

def test():
	im = cv2.imread("../../data/tc/area/1509_201507070000.tir.01.fld.png", 0)
	im_ref = cv2.imread("../../data/tc/area/1509_201507070010.tir.01.fld.png", 0)
	print im.shape, im_ref.shape

	vx, vy = motionEst(im, im_ref, (8, 8), 8)
	print vx.shape, vy.shape

	print vx

if __name__ == '__main__':
	test()