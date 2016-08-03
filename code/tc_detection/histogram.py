import numpy as np

def caclcHOG(velX, velY, num_orient):
	""" Calculate the Histogram of Oriented AMV/Gradient
		The angle is positive number in range of [0, 180]
		Params:
			velX, velY: (M, N) ndarray
				Input AMV/Gradients image of x-axis and y-axis resp
			num_orient: 
				Number of orientation bins
	"""
	height, width = velX.shape

	# calculate absolute magnitude of the AMV/Gradient
	magnitude = np.sqrt(velX ** 2 + velY ** 2)

	# set up histogram
	hist = np.zeros((height, width, num_orient))

	# calc cell orientation histogram
	bin_w = 180 / num_orient
	for y in range(height):
		for x in range(width):
			angle = (180 / np.pi) * (np.arctan2(velY[y, x], velX[y, x]) % np.pi)

			""" Calc orientation using bilinear interpolation
			"""

			# major bin
			m_bin = int(angle  / bin_w)
			if m_bin >= num_orient:
				print m_bin
				m_bin = num_orient - 1

			neig_offset = -1
			if angle > bin_w * (m_bin + 0.5):
				neig_offset = 1
			
			if m_bin == 0 and neig_offset == -1:
				hist[y, x, 8] = magnitude[y, x] * abs(angle - bin_w * (0 + 0.5)) / bin_w
				angle = 180 + angle
				hist[y, x, 0] = magnitude[y, x] * abs(angle - bin_w * (8 + 0.5)) / bin_w
			elif m_bin == 8 and neig_offset == 1:
				hist[y, x, 0] += magnitude[y, x] * abs(angle - bin_w * (8 + 0.5)) / bin_w
				angle = angle - 180
				hist[y, x, 8] += magnitude[y, x] * abs(angle - bin_w * (0 + 0.5)) / bin_w
			else:
				hist[y, x, m_bin] += magnitude[y, x] * abs(angle - bin_w * (m_bin + neig_offset + 0.5) ) / bin_w
				hist[y, x, m_bin + neig_offset] += magnitude[y, x] * abs(angle - bin_w * (m_bin + 0.5)) / bin_w

	return hist

""" Calculate integral image of Histogram of Oriented "Amotpheric Motion Vector"
"""
def calcIHO(velX, velY, num_orient, magnitude_threshold):
	hist = caclcHWD(velX, velY, num_orient, magnitude_threshold)
	
	height, width, _ = hist.shape

	# calc integral image of HOAMV/HOG
	integral_hist = np.copy(hist)
	for y in xrange(1, height):
		for x in xrange(1, width):
			for ang in range(num_orient):
				integral_hist[y, x, ang] += hist[y, x, ang] + \
											hist[y - 1, x, ang] + \
											hist[y, x - 1, ang] - \
											hist[y - 1, x - 1, ang]

	return integral_hist

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

