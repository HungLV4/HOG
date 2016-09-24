import sys
import os.path
import ntpath

import numpy as np

import cv2
import csv
import re
import math

import skimage

from skimage.feature import hog

# import gdal
# from gdalconst import *
# from osgeo import gdal_array, osr

# from _whoghistogram import whog_histograms

def calcGradient(im):
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

	return gx, gy

def hog_analysis(im, num_orientation):
	fd, hog_im = hog(im, orientations=num_orientation, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
	cv2.imwrite("results/1.png", hog_im)

def sdssahog(im, num_orientation):
	anm = np.zeros(im.shape, dtype=np.float64)
	intensity_anm = np.zeros(im.shape, dtype=np.float64)
	edge_anm = np.zeros(im.shape, dtype=np.float64)

	# calculate the gradient
	orientation_bin_width = 180. / num_orientation

	gx, gy = calcGradient(im)
	
	# calculate the histogram of orientation gradient for the whole image
	hist_orientation, magnitude, orientation = calcHOGDescriptor(gx, gy, 
		orientations=num_orientation, pixels_per_cell=im.shape, cells_per_block=(1, 1))
	hist_orientation /= hist_orientation.max()
	print hist_orientation

	# histogram of intensity
	intensity_bin_width = 10
	hist_intensity, _ = np.histogram(im, bins=np.arange(0, 255 + 2 * intensity_bin_width, intensity_bin_width))
	hist_intensity = hist_intensity / float(im.size)

	for y in range((im.shape)[0]):
		for x in range((im.shape)[1]):
			# for intensity abnormality
			index = int(im[y, x] / intensity_bin_width)
			intensity_abnormal[y, x] += 1 / (hist_intensity[index])

			# for edge orientation abnormality
			orientation_index = round(orientation[y, x] / orientation_bin_width) % num_orientation
			edge_orientation_abnormal[y, x] = hist_orientation[int(orientation_index)] * magnitude[y, x]

	# edge_orientation_abnormal = edge_orientation_abnormal / edge_orientation_abnormal.max()
	# intensity_abnormal = intensity_abnormal / intensity_abnormal.max()

	anm = edge_anm + intensity_anm
	gdal_array.SaveArray(anm, 'results/abnormal.tif', "GTiff")
	
	return abnormal

def thresholdAnomaly(rxd):
	threshold = 10
	
	anomaly = np.zeros(rxd.shape, dtype=np.uint8)
	indices = rxd > threshold
	anomaly[indices] = 255
	
	return anomaly

def segmentation(anomaly, vis):
	# noise removal
	kernel = np.ones((3, 3), np.uint8)
	opening = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, kernel, iterations = 2)

	# connected compponents
	kernel = np.ones((3, 3), np.uint8)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 2)

	temp = closing.copy()
	_, contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL , 2)
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		size =  rect[1]

		x, y, w, h = cv2.boundingRect(cnt)

		vis = cv2.drawContours(vis, [box], 0, (0, 0, 255), 1)

def test(im, index):
	gx, gy = calcGradient(im)
	
	glac = np.zeros((im.shape))
	hist = np.zeros((im.shape[0], im.shape[1], 2))
	whog_histograms(gx, gy, 9, im.shape[1], im.shape[0], 0.001, hist, glac)

	gdal_array.SaveArray(glac, 'results/rxd_VNR_PAN20150902.tif', "GTiff")

	anomaly = thresholdAnomaly(glac)

	gdal_array.SaveArray(anomaly, 'results/ano_VNR_PAN20150902.tif', "GTiff")

	segmentation(anomaly, im)

	cv2.imwrite('results/vis_VNR_PAN20150902.png', im)

if __name__ == '__main__':
	impath = "data/1.png"
	if os.path.isfile(impath):
		im = cv2.imread(impath, 0)
		hog_analysis(im, 9)
		
		# im = im.astype('float')
		# test(im, 3)