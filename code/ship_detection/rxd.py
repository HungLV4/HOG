import numpy as np
from scipy import signal
import cv2
from operator import itemgetter
from math import atan2, cos, pow, atan, floor
from numpy.linalg import inv

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

def calc_rxd(data, bands, size_column, size_row):
	if data.shape[0] < bands.max():
		print "Error: Number of band is larger then actual"
		return None

	num_bands = len(bands)
	num_pixels = size_column * size_row

	print num_bands, num_pixels
	if num_bands > 1:
		# reshaping
		GG = np.zeros((num_bands, num_pixels))
		for i in range(num_bands):
			band = bands[i]
			GG[i, ] = data[band].reshape(num_pixels)

		# calculating covariance matrix
		M = np.cov(GG)
		M_i = inv(M)

		print M, "\n", M_i

		avg = [np.mean(GG[i]) for i in range(num_bands)]

		rxd = np.zeros(num_pixels)
		for i in range(num_pixels):
			signal = GG[:, i] - avg
			rxd[i] = np.dot(np.dot(signal, M_i), signal.transpose())

		return rxd

	return None

if __name__ == '__main__':
	filepath = "data/crop/D6/VNR20150902_0_PXS.tif"

	dataset = gdal.Open(filepath, GA_ReadOnly)
	size_column = dataset.RasterXSize
	size_row = dataset.RasterYSize
	num_bands = dataset.RasterCount

	data = np.zeros((num_bands, size_row, size_column))
	for i in range(1, num_bands + 1):
		band = dataset.GetRasterBand(i)
		data[i - 1,:,:] = band.ReadAsArray(0, 0, size_column, size_row).astype(np.int)

	bands = np.arange(data.shape[0])
	rxd = calc_rxd(data, bands, size_column, size_row)

	result = rxd.reshape((size_row, size_column))
	gdal_array.SaveArray(result, 'results/D6/VNR20150902.tif', "GTiff")