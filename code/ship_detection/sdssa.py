import os
import numpy as np
import cv2

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _sdssa import integral_calc, texture_abnormal_calc
from _hough import ll_angle

from suppression import calc_inhibition

import csv
import subprocess

NOTDEF = -1

def crop_by_xy(scene_name):
	offset_x = 256
	offset_y = 256

	csv_path = "data/%s.csv" % scene_name
	img_path = "data/ori/%s.tif" % scene_name
	with open(csv_path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		index = 0
		for row in reader:
			x = int(row[0])
			y = int(row[1])
			out_path = "data/crop/%s_%d.tif" % (scene_name, index)
			subprocess.call("gdal_translate -srcwin %d %d %d %d -of Gtiff %s %s" % (x, y, offset_x, offset_y, img_path, out_path), shell=True)

			index += 1

def calc_sdssa_intensity(data, hist, binwidth, size_row, size_column):
	intensity_abnormal = np.zeros((size_row, size_column), dtype=np.float64)
	for i in range(size_row):
		for j in range(size_column):
			if data[i, j] > 0 and hist[(data[i, j] - 1) / binwidth] > 0:
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

def process(filename, wmode = 0):
	"""
	Input:
		filename: input file index
		wmode: 
			0: SDSSA weight
			1: Suppression weight
			otherwise: non-weigth
	"""
	# reading data
	filepath = "data/crop/%s.tif" % filename
	dataset = gdal.Open(filepath, GA_ReadOnly)
	
	size_column = dataset.RasterXSize
	size_row = dataset.RasterYSize
	
	band = dataset.GetRasterBand(1)
	data = band.ReadAsArray(0, 0, size_column, size_row).astype(np.int)

	# total_pixels = size_row * size_column
	total_pixels = sum(sum(i > 0 for i in data))

	# Calculate the histogram of image
	binwidth = 1
	hist, bin_edges = np.histogram(data, bins=np.arange(1, np.max(data) + 2 * binwidth, binwidth))

	# Calculate sdssa intensity abnormal
	intensity_abnormal = calc_sdssa_intensity(data, hist, binwidth, size_row, size_column)
	intensity_abnormal = intensity_abnormal / np.max(intensity_abnormal)
	
	if wmode != 1:
		# Calculate sdssa texture abnormal
		texture_abnormal = calc_sdssa_texture(data, size_row, size_column)
		texture_abnormal = texture_abnormal / np.max(texture_abnormal)

		# Calculate weight
		Cm, Ce, Cd = calc_sdssa_weight(hist, total_pixels)
		print Cm, Ce, Cd
		
		if wmode == 0:
			texture_abnormal = Cd * texture_abnormal
			intensity_abnormal = intensity_abnormal * (1 - Cd)
		
		gdal_array.SaveArray(texture_abnormal + intensity_abnormal, 'results/%s_m%d.tif' % (filename, wmode), "GTiff")
	elif wmode == 1:
		# Calculate gradient
		modgrad, modang = ll_angle(data, size_column, size_row, NOTDEF, 0)
		modgrad = modgrad / np.max(modgrad)

		# Calculate weight
		slope = calc_suppression_weight(modgrad, modang, size_row, size_column, 1)
		# gdal_array.SaveArray(slope, 'data/ship%d_slope.tif' % (index), "GTiff")
		
		modgrad = slope * modgrad
		intensity_abnormal = intensity_abnormal * (1 - slope)
		gdal_array.SaveArray(modgrad + intensity_abnormal, 'data/ship%d_%d.tif' % (index, wmode), "GTiff")

def process_by_scene(scene_name):
	csv_path = "data/%s.csv" % scene_name
	with open(csv_path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		data = list(reader)
		for index in range(len(data)):
			filename = "%s_%d" % (scene_name, index)
			process(filename)

if __name__ == '__main__':
	filelist = ["VNR20150117_PAN", "VNR20150202_PAN", "VNR20150303_PAN", "VNR20150609_PAN"]
	# filelist = ["VNR20150609_PAN"]
	for scene_name in filelist:
		# crop_by_xy(scene_name)
		process_by_scene(scene_name)
