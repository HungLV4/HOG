import os
import numpy as np
import cv2

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _motionEstBM import motionEstTSS
from _sdssa import integral_calc, texture_abnormal_calc

def sdssa():
	index = 6

	P1 = 0.9
	P2 = 0.01

	filepath = "results\%d.tif" % index
	dataset = gdal.Open(filepath, GA_ReadOnly)
	
	size_column = dataset.RasterXSize
	size_row = dataset.RasterYSize
	total_pixels = size_row * size_column

	band = dataset.GetRasterBand(1)
	data = band.ReadAsArray(0, 0, size_column, size_row).astype(np.int)

	"""
	Calculate weight values
	"""
	Cm = Ce = Cd = -1
	
	binwidth = 1
	hist, bin_edges = np.histogram(data, bins=np.arange(0, np.max(data) + binwidth, binwidth))
	
	sorted_hist = np.sort(hist)	
	cumm = 0.0
	for i in range(len(sorted_hist)):
		cumm += sorted_hist[len(sorted_hist) - 1 - i]
		if cumm / total_pixels >= P1 and Cm == -1:
			Cm = i
		if cumm / total_pixels >= 1 - P2 and Ce == -1:
			Ce = i
	Cd = float(Cm) / Ce

	"""
	Calculate texture abnormal
	"""
	# calculate integral image
	integral = np.zeros((size_row, size_column), dtype=np.float64)
	integral_sqr = np.zeros((size_row, size_column), dtype=np.float64)
	integral_calc(data, size_column, size_row, integral, integral_sqr)

	# 
	texture_abnormal = np.zeros((size_row, size_column), dtype=np.float32)
	texture_abnormal_calc(data, size_column, size_row, 2, 2, integral, integral_sqr, texture_abnormal)
	texture_abnormal = texture_abnormal / np.max(texture_abnormal)
	texture_abnormal = Cd * texture_abnormal

	"""
	Calculate intensity abnormal
	"""
	intensity_abnormal = np.zeros((size_row, size_column), dtype=np.float64)
	for i in range(size_row):
		for j in range(size_column):
			if hist[data[i, j] - 1] > 0:
				intensity_abnormal[i, j] = 1 / float(hist[data[i, j] - 1])

	intensity_abnormal = intensity_abnormal / np.max(intensity_abnormal)
	intensity_abnormal = intensity_abnormal * (1 - Cd)

	gdal_array.SaveArray(texture_abnormal, 'results/texture%d.tif' % index, "GTiff")
	gdal_array.SaveArray(intensity_abnormal, 'results/intensity%d.tif' % index, "GTiff")
	gdal_array.SaveArray(texture_abnormal + intensity_abnormal, 'results/abnormality%d.tif' % index, "GTiff")

if __name__ == '__main__':
	sdssa()
