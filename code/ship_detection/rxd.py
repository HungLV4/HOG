import numpy as np
from scipy import signal
import cv2
from operator import itemgetter
from math import atan2, cos, pow, atan, floor
from numpy.linalg import inv

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

def generate_fake_hs(data, size_column, size_row, ws = 5):
	"""
	Generate fake hyperspectral image using Shi et al 2014
	Input:
		data: 1-band data
		ws: window size
	Ouput:
		fake-multi-band data
	"""
	output = np.zeros((ws ** 2, size_row, size_column), dtype=np.int)
	for x in range(size_row):
		for y in range(size_column):
			for i in range(- ws / 2, ws / 2 + 1):
				for j in range(- ws / 2, ws / 2 + 1):
					if x + i >= 0 and x + i < size_row and y + j >= 0 and y + j < size_column:
						output[(i + ws / 2) * ws + (j + ws / 2), x, y] = data[x + i, y + j]
	return output

def calc_rxd(data, bands, size_column, size_row):
	if data.shape[0] < bands.max():
		print "Error: Number of band is larger then actual"
		return None

	num_bands = len(bands)
	num_pixels = size_column * size_row
	rxd = np.zeros(num_pixels)

	if num_bands > 1:
		# reshaping
		GG = np.zeros((num_bands, num_pixels))
		for i in range(num_bands):
			band = bands[i]
			GG[i, ] = data[band].reshape(num_pixels)

		# calculating covariance matrix
		M = np.cov(GG)
		M_i = inv(M)

		avg = [np.mean(GG[i]) for i in range(num_bands)]

		for i in range(num_pixels):
			signal = GG[:, i] - avg
			rxd[i] = np.dot(np.dot(signal, M_i), signal.transpose())
	elif num_bands == 1:
		band = bands[0]
		avg = np.mean(data[band])
		std = np.std(data[band])
		GG = data[band].reshape(num_pixels)
		for i in range(num_pixels):
			rxd[i] = abs((GG[i] - avg) / std)

	result = rxd.reshape((size_row, size_column))
	return result

if __name__ == '__main__':
	filepath = "data/crop/D6/VNR20150902_0_PAN.tif"

	dataset = gdal.Open(filepath, GA_ReadOnly)
	size_column = dataset.RasterXSize
	size_row = dataset.RasterYSize
	num_bands = dataset.RasterCount

	data = np.zeros((num_bands, size_row, size_column))
	for i in range(1, num_bands + 1):
		band = dataset.GetRasterBand(i)
		data[i - 1,:,:] = band.ReadAsArray(0, 0, size_column, size_row).astype(np.int)

	# bands = np.arange(data.shape[0])
	# bands = np.array([3])
	bands = np.array([0])

	rxd = calc_rxd(data, bands, size_column, size_row)

	gdal_array.SaveArray(rxd, 'results/D6/VNR20150902_PAN.tif', "GTiff")