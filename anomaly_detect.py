import gdal
from gdalconst import *
from osgeo import gdal_array, osr
import numpy as np
import math

def thresholdAnomaly(rxd, rows, cols):
	# finding threshold
	threshold = 1

	binwidth = 0.01
	
	# minval = rxd[rxd > 0].min()
	hist, bin_edges = np.histogram(rxd, bins=np.arange(0, np.max(rxd) + binwidth, binwidth))

	pixels = sum(hist)
	total = 0
	for i in range(len(hist) - 1, 0, -1):
		total += hist[i]
		if float(total) / pixels * 100.00 >= 4:
			threshold = bin_edges[i]
			break
	print "RXD threshold:", threshold
	# end finding threshold
	
	anomaly = np.zeros((rows, cols), dtype=np.uint8)
	indices = rxd > threshold
	anomaly[indices] = 255
	
	return anomaly

def calculateRXD(band, rows, cols):
	# calculate global covariance matrix
	# n/a
	# end calculate global covariance matrix
	
	rxd = np.zeros((rows, cols))
	tRXD = np.zeros((rows, cols))
	iRXD = np.zeros((rows, cols))

	sub_rect_cols = 128
	sub_rect_rows = 128
	window = 2
	
	x = 0
	while x < rows - 1:
		x1 = x + sub_rect_rows if x + sub_rect_rows <= rows - 1 else rows - 1

		y = 0
		while y < cols - 1:
			y1 = y + sub_rect_cols if y + sub_rect_cols <= cols - 1 else cols - 1

			brow = x - window if x - window >= 0 else 0
			trow = x1 + window if x1 + window < rows - 1 else rows - 1
			lcol = y - window if y - window >= 0 else 0
			rcol = y1 + window if y1 + window < cols - 1 else cols - 1

			drows = trow - brow + 1
			dcols = rcol - lcol + 1

			data = band.ReadAsArray(lcol, brow,
				dcols, drows).astype(np.int)

			dev = np.std(data)
			if dev >= 3:
				n = 4 * window * window
				sub_total = (x1 - x + 1) * (y1 - y + 1)

				# calculate histogram
				binwidth = 20
				hist, bin_edges = np.histogram(data, bins=np.arange(0, np.max(data) + 2 * binwidth, binwidth))

				# calculate integral image
				integral = np.zeros((drows, dcols))
				integral_sqr = np.zeros((drows, dcols))
				for i in range(drows):
					for j in range(dcols):
						integral[i, j] = data[i, j] + (integral[i - 1, j] if i - 1 >= 0 else 0)\
											+ (integral[i, j - 1] if j - 1 >= 0 else 0)\
											- (integral[i - 1, j - 1] if (j - 1 >= 0 and i - 1 >= 0) else 0)

						integral_sqr[i, j] = data[i, j] ** 2 + (integral_sqr[i - 1, j] if i - 1 >= 0 else 0)\
											+ (integral_sqr[i, j - 1] if j - 1 >= 0 else 0)\
											- (integral_sqr[i - 1, j - 1] if (j - 1 >= 0 and i - 1 >= 0) else 0)

				# calculate rxd
				for i in xrange(x, x1):
					for j in xrange(y, y1):
						signal = data[i - brow, j - lcol]
						if signal == 0:
							continue

						# get mean of neighborhood pixels data
						bnrow = i - window - brow
						bnrow = bnrow if bnrow > 0 else 0

						tnrow = (i + window if i + window <= trow else trow) - brow
						
						lncol = j - window - lcol
						lncol = lncol if lncol > 0 else 0

						rncol = (j + window if j + window <= rcol else rcol) - lcol

						if data[bnrow, lncol] == 0 or data[bnrow, rncol] == 0 or data[tnrow, lncol] == 0 or data[tnrow, rncol] == 0:
							continue

						S1 = integral[tnrow, rncol] + integral[bnrow, lncol]\
								- integral[bnrow, rncol] - integral[tnrow, lncol]

						S2 = integral_sqr[tnrow, rncol] + integral_sqr[bnrow, lncol]\
								- integral_sqr[bnrow, rncol] - integral_sqr[tnrow, lncol]

						tRXD[i, j] = (math.sqrt(n * S2 - (S1 ** 2)) / S1) if S1 > 0 else 0

						index = int(signal / binwidth)
						iRXD[i, j] = (sub_total / (hist[index] * 100)) if (hist[index]) > 0 else 1
			y = y1
		x = x1
	rxd = iRXD / np.max(iRXD) + tRXD / np.max(tRXD)
	return rxd
