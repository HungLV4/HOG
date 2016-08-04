import numpy as np


""" Calculate the Histogram of Wind Speed
"""
def calcHWS(velX, velY, num_bin):
	height, width = velX.shape

	# calculate absolute magnitude of the AMV/Gradient
	magnitude = np.sqrt(velX ** 2 + velY ** 2)

	hist = np.zeros(num_bin + 1)
	for y in range(height):
		for x in range(width):
			b = int(magnitude[y, x]) if int(magnitude[y, x]) < num_bin else num_bin
			hist[b] += 1
	return hist

""" Calculate the Histogram of Direction/Speed Ratio
"""
def calcDSR(velX, velY, amv_threshold):
	height, width = velX.shape

	dsr = np.zeros((height, width))

	# calculate absolute magnitude of the AMV/Gradient
	magnitude = np.sqrt(velX ** 2 + velY ** 2)
	for y in range(height):
		for x in range(width):
			if magnitude[y, x] >= amv_threshold:
				angle = (180 / np.pi) * (np.arctan2(velY[y, x], velX[y, x]) % np.pi)
				dsr[y, x] = angle / magnitude[y, x]
	
	# calculate the histogram
	# binwidth = 10
	# hist, bin_edges = np.histogram(data, bins=np.arange(1, np.max(data) + 2 * binwidth, binwidth))
	# print dsr.min(), dsr.max()
	
	# return hist