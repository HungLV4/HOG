import numpy as np
import cv2
from operator import itemgetter

import gdal
from gdalconst import *
from osgeo import gdal_array, osr

from _hough import HoughLinesStandard

index = 4

img = cv2.imread("hough/hough%d.png" % index)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""
OpenCV Hough
"""
# edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
# lines = cv2.HoughLines(edges, 1, np.pi/180, 15)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img, (x1, y1), (x2, y2),(255, 0, 255), 1)
# cv2.imwrite('hough/houghlines.jpg', img)

""" 
Proposed Hough transform
"""
mode = 1 # 0: Hough , otherwise: Weighted Hough
maxline = 20
lines, accum, edges = HoughLinesStandard(gray, gray.shape[1], gray.shape[0], 1, np.pi / 180, 100, 0, np.pi, maxline, mode)
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

gdal_array.SaveArray(edges, 'hough/edges%d.tif' % index, "GTiff")