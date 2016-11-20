import os
import numpy as np
import cv2

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _sdssa import integral_calc, texture_abnormal_calc
from _hough import ll_angle, hough_lines
from _rxd import generate_fake_hs

from suppression import calc_inhibition
from rxd import calc_rxd
from otsu import calc_otsu

import csv
import subprocess

NOTDEF = -1

def crop_by_shp(scene_name):
	shp_path = "../../../../../../data/ROI_SHP/CSG.shp"
	img_path = "../../../../../../data/VNREDSat/%s/%s_PAN.tif" % (scene_name, scene_name)
	out_path = "data/ori/CSG/%s_PAN.tif" % scene_name
	
	subprocess.call("gdalwarp -q -cutline %s -crop_to_cutline -of GTiff %s %s" % (shp_path, img_path, out_path), shell=True)

def crop_by_xy(scene_name):
	offset_x = 256
	offset_y = 256

	csv_path = "data/csv/%s.csv" % scene_name
	img_path = "data/ori/CSG/%s_PXS.tif" % scene_name
	with open(csv_path, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		index = 0
		for row in reader:
			x = int(row[0]) - 1
			y = int(row[1]) - 1
			d = int(row[2])

			if d == 1 or d == 2 or d == 3 or d  == 6:
				print scene_name, index
				out_path = "data/crop/D%d/%s_%d_PXS.tif" % (d, scene_name, index)
				subprocess.call("gdal_translate -srcwin %d %d %d %d -of Gtiff %s %s" % (x, y, offset_x, offset_y, img_path, out_path), shell=True)

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

def calc_threshold(abnormal, size_column, size_row, mode):
	"""
	Fiding threshold using Otsu method
	"""
	if mode == 'otsu':
		return calc_otsu(abnormal, size_column, size_row)
	
	return 0

def binaritify(abnormal, threshold):
	"""
	Binarize image using single threshold value
	"""
	binary_img = np.zeros(abnormal.shape, dtype=np.uint8)
	binary_img[abnormal > threshold] = 255
	return binary_img

def calc_ssa(data, size_column, size_row, abnormal, wmode):
	"""
	Input:
		data: input data file
		wmode: 
			0: SDSSA weight
			1: Suppression weight
	Output:
		abnormal: abnormal calculated
	"""
	binwidth = 1
	NOTDEF = 0

	total_pixels = sum(sum(i > 0 for i in data))

	# Calculate the histogram of image
	hist, bin_edges = np.histogram(data, bins=np.arange(1, np.max(data) + 2 * binwidth, binwidth))

	# Calculate sdssa intensity abnormal
	intensity_abnormal = calc_sdssa_intensity(data, hist, binwidth, size_row, size_column)
	
	if wmode == 0:
		# Calculate sdssa texture abnormal
		texture_abnormal = calc_sdssa_texture(data, size_row, size_column)

		# Calculate weight
		Cm, Ce = calc_sdssa_weight(hist, total_pixels)
		Cd = float(Cm) / Ce

		# add weigth
		texture_abnormal = Cd * texture_abnormal
		intensity_abnormal = intensity_abnormal * (1 - Cd)
		
		# calc final abnormal and thresholding
		abnormal = texture_abnormal + intensity_abnormal
		return abnormal, Cd
	else:
		# Calculate gradient
		modgrad, modang = ll_angle(data, size_column, size_row, NOTDEF, 0)
		modgrad = modgrad / np.max(modgrad)

		# Calculate weight
		contours, slope = calc_suppression_weight(modgrad, modang, size_row, size_column, 3)
		
		# add weigth
		modgrad = slope * modgrad
		intensity_abnormal = intensity_abnormal * (1 - slope)
		
		# calc final abnormal and thresholding
		abnormal = modgrad + intensity_abnormal
		return abnormal, slope

def calc_binary_img(data, band_idx, size_column, size_row, wmode):
	# asset
	if (wmode == 0 or wmode == 1) and len(band_idx) > 1:
		print "Error: SSA modes only support 1-band data"
		return None
	
	abnormal = np.zeros((size_row, size_column))
	binary_img = np.zeros(abnormal.shape, dtype=np.uint8)

	if wmode == 0 and band_idx[0] <= data.shape[0]:
		# SSA using Guang Yang  et al 2014 weight
		print "Calculating using SSA"
		abnormal, Cd = calc_ssa(data[band_idx[0], :, :], size_column, size_row, abnormal, wmode)
	elif wmode == 1 and band_idx[0] <= data.shape[0]:
		# SSA using surround suppression weight
		print "Calculating using SST"
		abnormal, slope = calc_ssa(data[band_idx[0],:,:], size_column, size_row, abnormal, wmode)
	elif wmode == 2:
		# RXD using multispectral
		print "Calculating using RXD"
		abnormal = calc_rxd(data, band_idx, size_column, size_row)
	elif wmode == 3 and band_idx[0] <= data.shape[0]:
		# RXD using fake-multispectral
		print "Calculating using fake RXD"

		# generate fake hyperspectral
		ws = 5
		fakeHs = np.zeros((ws ** 2, size_row, size_column), dtype=np.int)
		generate_fake_hs(data[band_idx[0], :, :], size_column, size_row, ws, fakeHs)

		# RXD using fake multispectral
		abnormal = calc_rxd(fakeHs, np.arange(ws ** 2), size_column, size_row)

	if abnormal.max() == 0:
		return abnormal, binary_img

	# normalize abnotmal into [0, 1] range
	abnormal = abnormal / abnormal.max()

	threshold = calc_threshold(abnormal[2:-2,2:-2], size_column - 5, size_row - 5, mode='otsu')
	print threshold

	binary_img = binaritify(abnormal, threshold)

	# remove noise/small regions
	kernel = np.ones((2, 2), np.uint8)
	binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations = 2)

	kernel = np.ones((3, 3), np.uint8)
	binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations = 1)

	return abnormal, binary_img

def get_list_contours(binary_img):
	temp = binary_img.copy()
	_, contours, _ = cv2.findContours(temp, cv2.RETR_LIST, 2)
	
	rotated_bounding_boxs = []
	for cnt in contours:
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		rotated_bounding_boxs.append([box])

	return rotated_bounding_boxs

def draw_candidates(vis, contours, color):
	for cnt in contours:
		cv2.drawContours(vis, cnt, 0, color, 1)

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
	src = 0

	for wmo in range(0, 4):
		csv_path = "data/csv/%s.csv" % scene_name
		with open(csv_path, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			index = 0
			for row in reader:
				# reading data
				filename = "%s_%d" % (scene_name, index)
				d = int(row[2])
				if d == 1 or d == 2 or d == 3 or d == 6: 
					filepath = "data/crop/D%d/%s_PAN.tif" % (d, filename)
					print "Processing", filepath
					
					# preparing result folders
					directory = 'results/D%d/%s' % (d, filename)
					if not os.path.exists(directory):
						os.makedirs(directory)
					
					# reading metadata
					dataset = gdal.Open(filepath, GA_ReadOnly)
					size_column = dataset.RasterXSize
					size_row = dataset.RasterYSize
					size_band = dataset.RasterCount

					# reading data into n-dimension array
					data = np.zeros((size_band, size_row, size_column), dtype=np.int)
					for i in range(1, size_band + 1):
						band = dataset.GetRasterBand(i)
						data[i - 1, :, :] = band.ReadAsArray(0, 0, size_column, size_row).astype(np.int)

					""" 
						Stage 1: coarse candidates selection using abnormality threshold
					"""
					band_idx = np.array([0])
					abnormal, binary_img = calc_binary_img(data, band_idx, size_column, size_row, wmode)
					
					gdal_array.SaveArray(abnormal, 'results/D%d/%s/%d_%d.tif' % (d, filename, wmode, src), "GTiff")
					cv2.imwrite('results/D%d/%s/%d_%d.png' % (d, filename, wmode, src), binary_img)

					# rbbs = get_list_contours(binary_img)

					"""
						Stage 2: refine candidates using shape information
					"""
					# refine_segment(data, rbbs)

					# vis = cv2.imread("data/D%d/%s.png" % (d, filename))
					# vis = draw_candidates(vis, rbbs, (0, 0, 255))
					# cv2.imwrite('results/D%d/%s.png' % (d, filename), vis)

				index += 1

if __name__ == '__main__':
	filelist = ["VNR20150117", "VNR20150202",
				"VNR20150303", "VNR20150417",
				"VNR20150508", "VNR20150609",
				"VNR20150726", "VNR20150816",
				"VNR20150904",
				"VNR20150415", "VNR20150628", "VNR20150902"]
	
	filelist = ["VNR20150117", "VNR20150202",
				"VNR20150303", "VNR20150417",
				"VNR20150508", "VNR20150609",
				"VNR20150726", "VNR20150816",
				"VNR20150904",
				"VNR20150415", "VNR20150628", "VNR20150902"]
	
	for scene_name in filelist:
		# crop_by_shp(scene_name)
		# crop_by_xy(scene_name)
		process_by_scene(scene_name)