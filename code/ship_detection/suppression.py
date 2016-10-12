import numpy as np
from scipy import signal
import cv2
from operator import itemgetter
from math import atan2, cos, pow, atan, floor

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _hough import hough_lines, ll_angle
from _anisotropic import calc_anisotropic_inhibterm

def gaussian2d(n, sigma):
	k1d = cv2.getGaussianKernel(n, sigma)
	k1d = k1d.reshape(n)

	k2d = np.zeros((n, n))
	for i in range(n):
		k2d[i, :] = k1d[i] * k1d

	return k2d

def diffOfGausskernel2d(sigma, k1, k2):
	# calculate kernel size
	n = int(floor(sigma) * (3 * k2 + k1) - 1)

	return gaussian2d(n, k2 * sigma) - gaussian2d(n, k1 * sigma)

def inhibkernel2D(sigma, k1, k2):
	diff_of_gauss = diffOfGausskernel2d(sigma, k1, k2)
	# set every negative values to 0 (H function)
	diff_of_gauss[diff_of_gauss < 0] = 0

	norm_L1 = sum(sum(abs(diff_of_gauss)))
	if norm_L1 != 0:
		return diff_of_gauss / norm_L1
	else:
		return 0

def calc_inhibition(modgrad, modang, sigma, alpha, k1, k2, rows, cols, mode = 1):
	w = inhibkernel2D(sigma, k1, k2)
	
	term = calc_anisotropic_inhibterm(modgrad, w, modang, cols, rows, w.shape[1], w.shape[0], mode)
	contour = abs(modgrad) - alpha * term
	contour[contour < 0] = 0 # set every negative value to 0 (H-function)

	slope = np.zeros((rows, cols))
	for i in range(rows):
		for j in range(cols):
			if modgrad[i, j] > 0:
				slope[i, j] = atan(term[i, j] / modgrad[i, j])

	return contour, term, slope

if __name__ == '__main__':
	NOTDEF = 0
	sigma = 1.5
	k1 = 1
	k2 = 4
	alpha = 2
	
	# mode: 0: isotropic, 1: anisotropic, 2: invert anisotropic
	mode = 0
	
	index = 7

	# read image
	img = cv2.imread("data/ship%d.png" %index)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rows, cols = gray.shape[0], gray.shape[1]

	# calculate the gradient magnitude and orientation
	modgrad, modang = ll_angle(gray, cols, rows, NOTDEF)
	
	# calculation of the surround inhibition and inhibition term
	filtered, term, slope = calc_inhibition(modgrad, modang, sigma, alpha, k1, k2, rows, cols, mode)
	# gdal_array.SaveArray(filtered, 'data/ship%d_%d.tif' % (index, mode), "GTiff")
	gdal_array.SaveArray(slope, 'data/ship%d_s.tif' % index, "GTiff")
	
	# gdal_array.SaveArray(modgrad, 'data/ship%d_e.tif' % index, "GTiff")




