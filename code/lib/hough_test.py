import numpy as np
from scipy import signal
import cv2
from operator import itemgetter
from math import atan2, cos, pow, atan

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _hough import hough_lines, ll_angle

index = 9

NOTDEF = -1

img = cv2.imread("hough/hough%d.png" % index)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape[0], gray.shape[1]

""" Calculating gradient angle and magnitude
"""
# modgrad, modang = ll_angle(gray, cols, rows, NOTDEF)
modgrad = cv2.Canny(gray, 100, 200)
modgrad = modgrad.astype(np.float32)
modgrad[modgrad < 255] = NOTDEF

""" Surrounding suprression
"""
sigma = 1.5
kernel_size = 25
alpha = 10

# Get 1d Gaussian kernel
k1d_1sig = cv2.getGaussianKernel(kernel_size, sigma)
k1d_1sig = k1d_1sig.reshape(kernel_size)

k1d_4sig = cv2.getGaussianKernel(kernel_size, 4 * sigma)
k1d_4sig = k1d_4sig.reshape(kernel_size)

# Generate 2d kernel (no lib support 2d kernel generation)
k2d_1sig = np.zeros((kernel_size, kernel_size))
k2d_4sig = np.zeros((kernel_size, kernel_size))
for i in range(kernel_size):
	k2d_1sig[i, :] = k1d_1sig[i] * k1d_1sig
	k2d_4sig[i, :] = k1d_4sig[i] * k1d_4sig

# Calculate the DoG-
DoG =  k2d_4sig - k2d_1sig
blob = signal.convolve2d(modgrad, DoG, mode='same')
gdal_array.SaveArray(blob, 'hough/blob%d.tif' % index, "GTiff")

DoG_inverse = DoG
DoG_inverse[DoG_inverse > 0] = 0

# Calculate the convolution of edge with DoG_inverse
T3 = signal.convolve2d(modgrad, DoG_inverse, mode='same')

# Calculate the covolution of edge with kernel 2d of 1 sigma
# by convolve it in each direction
temp_T2 = None
for i in range(rows):
	if temp_T2 == None:
		temp_T2 = np.convolve(modgrad[i], k1d_1sig, mode='same')
	else:
		temp_T2 = np.row_stack((temp_T2, np.convolve(modgrad[i], k1d_1sig, mode='same')))

T2 = None
for i in range(cols):
	if T2 == None:
		T2 = np.convolve(temp_T2[:, i], k1d_1sig, mode='same')
	else:
		T2 = np.column_stack((T2, np.convolve(temp_T2[:, i], k1d_1sig, mode='same')))

# Calculate the covolution of edge with kernel 2d of 1 sigma
# by convolve it in each direction
temp_T1 = None
for i in range(rows):
	if temp_T1 == None:
		temp_T1 = np.convolve(modgrad[i], k1d_4sig, mode='same')
	else:
		temp_T1 = np.row_stack((temp_T1, np.convolve(modgrad[i], k1d_4sig, mode='same')))

T1 = None
for i in range(cols):
	if T1 == None:
		T1 = np.convolve(temp_T1[:, i], k1d_4sig, mode='same')
	else:
		T1 = np.column_stack((T1, np.convolve(temp_T1[:, i], k1d_4sig, mode='same')))

# calculate the final weight
weight = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
	for j in range(cols):
		weight[i, j] = pow(cos(atan((T1[i, j] - T2[i, j] - T3[i, j]) / modgrad[i, j])), 10)

""" Hough line transform
"""
mode = 1 # 0: Naive-Hough , otherwise: Weighted-Hough
maxline = 10
lines, accum = hough_lines(modgrad, weight, gray.shape[1], gray.shape[0], 1, np.pi / 180, 30, 0, np.pi, maxline, NOTDEF, mode)
for line in lines:
	rho, theta = line[0], line[1]

	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho
	
	x1 = int(x0 + 1000 * (-b))
	y1 = int(y0 + 1000 * (a))
	x2 = int(x0 - 1000 * (-b))
	y2 = int(y0 - 1000 * (a))

	cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
cv2.imwrite('hough/houghlines_%d.jpg' % index, img)
gdal_array.SaveArray(accum, 'hough/accum%d.tif' % index, "GTiff")

gdal_array.SaveArray(modgrad, 'hough/edges%d.tif' % index, "GTiff")
gdal_array.SaveArray(weight, 'hough/weight%d.tif' % index, "GTiff")