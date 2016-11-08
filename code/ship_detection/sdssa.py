import os
import numpy as np
import cv2

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _sdssa import integral_calc, texture_abnormal_calc
from _hough import ll_angle, hough_lines

from suppression import calc_inhibition

import csv
import subprocess

NOTDEF = -1

def crop_by_xy(scene_name):
	offset_x = 256
	offset_y = 256

	csv_path = "data/csv/%s.csv" % scene_name
	img_path = "data/ori/CSG/%s.png" % scene_name
	with open(csv_path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		index = 0
		for row in reader:
			print scene_name, index
			x = int(row[0]) - 1
			y = int(row[1]) - 1
			out_path = "data/crop/vis/%s_%d.png" % (scene_name, index)
			subprocess.call("gdal_translate -srcwin %d %d %d %d -of png %s %s" % (x, y, offset_x, offset_y, img_path, out_path), shell=True)

			index += 1

def calc_sdssa_intensity(data, hist, binwidth, size_row, size_column):
	intensity_abnormal = np.zeros((size_row, size_column), dtype=np.float64)
	for i in range(size_row):
		for j in range(size_column):
			if data[i, j] > 0 and hist[(data[i, j] - 1) / binwidth] > 0:
				intensity_abnormal[i, j] = 1 / float(hist[(data[i, j] - 1) / binwidth])
	return intensity_abnormal / np.max(intensity_abnormal)

def calc_sdssa_texture(data, size_row, size_column):
	# calculate integral image
	integral = np.zeros((size_row, size_column), dtype=np.float64)
	integral_sqr = np.zeros((size_row, size_column), dtype=np.float64)
	integral_calc(data, size_column, size_row, integral, integral_sqr)

	# 
	texture_abnormal = np.zeros((size_row, size_column), dtype=np.float32)
	texture_abnormal_calc(data, size_column, size_row, 2, 2, integral, integral_sqr, texture_abnormal)

	return texture_abnormal / np.max(texture_abnormal)

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
	return Cm, Ce

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

	contours, term, slope = calc_inhibition(modgrad, modang, sigma, alpha, k1, k2, rows, cols, mode)

	return contours, slope

def auto_threshold_sdssa(abnormal, weight):
	binary_img = np.zeros(abnormal.shape, dtype=np.uint8)
	binary_img[abnormal > weight] = 255

	return binary_img

def auto_threshold_supp(abnormal, slope):
	pass

def calc_binary_img(data, size_column, size_row, wmode = 0):
	"""
	Input:
		data: input data file
		wmode: 
			0: SDSSA weight
			1: Suppression weight
			otherwise: non-weigth
	Output:
		Binary mask
	"""
	binwidth = 1
	NOTDEF = 0

	total_pixels = sum(sum(i > 0 for i in data))
	
	"""
	Calculating abnormality
	"""
	abnormal = np.zeros((size_row, size_column))

	# Calculate the histogram of image
	hist, bin_edges = np.histogram(data, bins=np.arange(1, np.max(data) + 2 * binwidth, binwidth))

	# Calculate sdssa intensity abnormal
	intensity_abnormal = calc_sdssa_intensity(data, hist, binwidth, size_row, size_column)
	
	if wmode != 1:
		# Calculate sdssa texture abnormal
		texture_abnormal = calc_sdssa_texture(data, size_row, size_column)

		# Calculate weight
		Cm, Ce = calc_sdssa_weight(hist, total_pixels)
		Cd = float(Cm) / Ce

		if wmode == 0:
			texture_abnormal = Cd * texture_abnormal
			intensity_abnormal = intensity_abnormal * (1 - Cd)
		
		abnormal = texture_abnormal + intensity_abnormal
		binary_img = auto_threshold_sdssa(abnormal, Cd)
	elif wmode == 1:
		# Calculate gradient
		modgrad, modang = ll_angle(data, size_column, size_row, NOTDEF, 0)
		modgrad = modgrad / np.max(modgrad)

		# Calculate weight
		contours, slope = calc_suppression_weight(modgrad, modang, size_row, size_column, 3)
		
		modgrad = slope * modgrad
		intensity_abnormal = intensity_abnormal * (1 - slope)
		abnormal = modgrad + intensity_abnormal

	# remove noise/small regions
	kernel = np.ones((2, 2), np.uint8)
	binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations = 2)

	kernel = np.ones((3, 3), np.uint8)
	binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations = 2)

	return binary_img

def get_list_contours(binary_img):
	temp = binary_img.copy()
	_, contours, _ = cv2.findContours(temp, cv2.RETR_LIST, 2)
	
	rotated_bounding_boxs = []
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		rotated_bounding_boxs.append(box)

	return rotated_bounding_boxs

def draw_candidates(binary_img, contours, color):
	vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)		
	for cnt in contours:
		cv2.drawContours(vis, [box], 0, color, 1)

	return vis

def refine_segment(data, rbbs):
	"""
	Input:
		data: real image
		rbbs: list of rotated bouding box
	Ouput:
		new binary mask
	"""

	# edges detection with no threshold
	# edges, _ = ll_angle(binary_img.astype(int), size_column, size_row, NOTDEF, 0)
	# hmode = 0 # Naive Hough
	# maxline = 4
	# lines, hs = hough_lines(edges, None, size_column, size_row, 1, np.pi / 36, 20, 0, np.pi, maxline, NOTDEF, hmode)
	# for line in lines:
	# 	rho, theta = line[0], line[1]

	# 	a = np.cos(theta)
	# 	b = np.sin(theta)
	# 	x0 = a * rho
	# 	y0 = b * rho
		
	# 	x1 = int(x0 + 1000 * (-b))
	# 	y1 = int(y0 + 1000 * (a))
	# 	x2 = int(x0 - 1000 * (-b))
	# 	y2 = int(y0 - 1000 * (a))

	# 	cv2.line(binary_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
	# gdal_array.SaveArray(edges, 'results/%d/%s_e.tif' % (wmode, filename), "GTiff")
	# gdal_array.SaveArray(hs.astype(int), 'results/%d/%s_hs.tif' % (wmode, filename), "GTiff")
	pass

def process_by_scene(scene_name):
	wmode = 0 # binary image calculation mode

	csv_path = "data/%s.csv" % scene_name
	with open(csv_path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		data = list(reader)
		for index in range(len(data)):
			# reading data
			filename = "%s_%d" % (scene_name, index)
			print "Processing", filename
			filepath = "data/crop/%s.tif" % filename
			
			dataset = gdal.Open(filepath, GA_ReadOnly)
			size_column = dataset.RasterXSize
			size_row = dataset.RasterYSize

			band = dataset.GetRasterBand(1)
			data = band.ReadAsArray(0, 0, size_column, size_row).astype(np.int)
			
			""" 
				Stage 1: coarse candidates selection using abnormality threshold
			"""
			binary_img = calc_binary_img(data, size_column, size_row, wmode)
			rbbs = get_list_contours(binary_img)

			"""
				Stage 2: refine candidates using shape information
			"""
			refine_segment(data, rbbs)

			# vis = draw_candidates(binary_img, contours, (0, 0, 255))
			# cv2.imwrite('results/1/%d/%s.png' % (wmode, filename), vis)

if __name__ == '__main__':
	filelist = ["VNR20150117_PAN", "VNR20150202_PAN",
				"VNR20150303_PAN", "VNR20150417_PAN",
				"VNR20150508_PAN", "VNR20150609_PAN",
				"VNR20150726_PAN", "VNR20150816_PAN",
				"VNR20150904_PAN",
				"VNR20150117_PAN_IS", "VNR20150202_PAN_IS",
				"VNR20150303_PAN_IS", "VNR20150417_PAN_IS",
				"VNR20150508_PAN_IS", "VNR20150609_PAN_IS",
				"VNR20150726_PAN_IS", "VNR20150816_PAN_IS",
				"VNR20150904_PAN_IS",
				"VNR20150415_PAN", "VNR20150628_PAN", "VNR20150902_PAN"]
	
	filelist = ["VNR20150415_PAN", "VNR20150628_PAN", "VNR20150902_PAN"]
	
	for scene_name in filelist:
		# crop_by_xy(scene_name)
		process_by_scene(scene_name)