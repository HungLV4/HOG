import sys
import os.path
import ntpath

import numpy as np

import cv2
import csv
import re
import math

import skimage

from hog import calcHOGDescriptor
from skimage.feature import hog

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

def sdssahog(im, num_orientation):
	abnormal = np.zeros(im.shape, dtype=np.float64)
	intensity_abnormal = np.zeros(im.shape, dtype=np.float64)
	edge_orientation_abnormal = np.zeros(im.shape, dtype=np.float64)

	# calculate the gradient
	orientation_bin_width = 180. / num_orientation

	# gx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=-1)
	# gy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=-1)

	gx = np.empty(im.shape, dtype=np.double)
	gx[:, 0] = 0
	gx[:, -1] = 0
	gx[:, 1:-1] = im[:, 2:] - im[:, :-2]

	gy = np.empty(im.shape, dtype=np.double)
	gy[0, :] = 0
	gy[-1, :] = 0
	gy[1:-1, :] = im[2:, :] - im[:-2, :]
	
	hist_orientation = calcHOGDescriptor(gx, gy, orientations=num_orientation, pixels_per_cell=(32, 32), cells_per_block=(1, 1))

	magnitude = np.hypot(gx, gy)
	orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180

	# histogram of intensity
	intensity_bin_width = 10
	hist_intensity, _ = np.histogram(im, bins=np.arange(0, 255 + 2 * intensity_bin_width, intensity_bin_width))
	hist_intensity = hist_intensity / float(sum(hist_intensity))

	for y in range((im.shape)[0]):
		for x in range((im.shape)[1]):
			# for intensity abnormality
			intensity_index = int(im[y, x] / intensity_bin_width)
			intensity_abnormal[y, x] += 1 / (hist_intensity[intensity_index]) if hist_intensity[intensity_index] > 0 else 1

			# for edge orientation abnormality
			orientation_index = round(orientation[y, x] / orientation_bin_width) % num_orientation
			edge_orientation_abnormal[y, x] = hist_orientation[int(orientation_index)] * magnitude[y, x]

	edge_orientation_abnormal = edge_orientation_abnormal / edge_orientation_abnormal.max()
	intensity_abnormal = intensity_abnormal / intensity_abnormal.max()

	abnormal = edge_orientation_abnormal + intensity_abnormal
	
	return abnormal

if __name__ == '__main__':
	im = cv2.imread("../../train/ship/pos/64x128/1.png", 0)
	im = im.astype('float')

	abnormal = sdssahog(im, 9)

	gdal_array.SaveArray(abnormal, 'abnormal.tif', "GTiff")