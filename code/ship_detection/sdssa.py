import os
import numpy as np
import cv2

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _motionEstBM import motionEstTSS
from _sdssa import integral_calc, texture_abnormal_calc

from suppression import calc_inhibition

def calc_sdssa_intensity(data, hist, size_row, size_column):
	intensity_abnormal = np.zeros((size_row, size_column), dtype=np.float64)
	for i in range(size_row):
		for j in range(size_column):
			if hist[data[i, j] - 1] > 0:
				intensity_abnormal[i, j] = 1 / float(hist[data[i, j] - 1])
	return intensity_abnormal

def calc_sdssa_texture(data, size_row, size_column):
	# calculate integral image
	integral = np.zeros((size_row, size_column), dtype=np.float64)
	integral_sqr = np.zeros((size_row, size_column), dtype=np.float64)
	integral_calc(data, size_column, size_row, integral, integral_sqr)

	# 
	texture_abnormal = np.zeros((size_row, size_column), dtype=np.float32)
	texture_abnormal_calc(data, size_column, size_row, 2, 2, integral, integral_sqr, texture_abnormal)

	return texture_abnormal

def calc_sdssa_weight(hist, total_pixels):
	"""
	Calculate weight values
	"""
	P1 = 0.9
	P2 = 0.01

	Cm = Ce = Cd = -1
		
	sorted_hist = np.sort(hist)
	cumm = 0.0
	for i in range(len(sorted_hist)):
		cumm += sorted_hist[len(sorted_hist) - 1 - i]
		if cumm / total_pixels >= P1 and Cm == -1:
			Cm = i
		if cumm / total_pixels >= 1 - P2 and Ce == -1:
			Ce = i
	Cd = float(Cm) / Ce

	return Cm, Ce, Cd

def calc_suppression_weight(modgrad, modang, rows, cols, mode):
	"""
	Input: 
		modgrad: gradient magnitude
		modang: gradient orientation
		mode: 
			0: isotropic, 
			1: anisotropic, 
			2: invert anisotropic
	"""
	sigma = 1.5
	k1 = 1
	k2 = 4
	alpha = 2

	filtered, term, slope = calc_inhibition(modgrad, modang, sigma, alpha, k1, k2, rows, cols, mode)

	return slope

def sdssa():
	index = 6

	# reading data
	filepath = "data/%d.tif" % index
	dataset = gdal.Open(filepath, GA_ReadOnly)
	
	size_column = dataset.RasterXSize
	size_row = dataset.RasterYSize
	total_pixels = size_row * size_column

	band = dataset.GetRasterBand(1)
	data = band.ReadAsArray(0, 0, size_column, size_row).astype(np.int)

	# Calculate the histogram of image
	binwidth = 1
	hist, bin_edges = np.histogram(data, bins=np.arange(0, np.max(data) + binwidth, binwidth))

	# Calculate intensity abnormal
	intensity_abnormal = calc_sdssa_intensity(data, hist, size_row, size_column)

	# Calculate texture abnormal
	texture_abnormal = calc_sdssa_texture(data, size_row, size_column)
	
	# Normalizing
	intensity_abnormal = intensity_abnormal / np.max(intensity_abnormal)
	texture_abnormal = texture_abnormal / np.max(texture_abnormal)
	
	# Calculate the weighting values
	Cm, Ce, Cd = calc_sdssa_weight(hist, total_pixels)
	slope = calc_suppression_weight(texture_abnormal, None, size_row, size_column, 0)

	# Adding weight
	# texture_abnormal = slope * texture_abnormal
	# intensity_abnormal = intensity_abnormal * (1 - slope)
	
	gdal_array.SaveArray(texture_abnormal * slope + intensity_abnormal * (1 - slope), 'data/suppression%d.tif' % index, "GTiff")
	gdal_array.SaveArray(texture_abnormal * Cd + intensity_abnormal * (1 - Cd), 'data/sdssa%d.tif' % index, "GTiff")

if __name__ == '__main__':
	sdssa()
