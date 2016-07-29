import cv2
import numpy as np
import itertools
import math

class HOGDetector(object):
	"""docstring for HOGDetector"""
	def __init__(self):
		super(HOGDetector, self).__init__()
		
	"""
		Calculate integral gradient of image (w, h, numorient)
	          
	    returns 
	        Integral gradient image
	 
	    params 
	        im: original image
	        numorient: number of orientation bins, default is 9 (-4..4)
	    """
	def _calc_weighted_gradient(self):    
	    # calculate gradient in 2 direction
		gx = cv2.Sobel(self.im, cv2.CV_64F, 1, 0, ksize=3)
		gy = cv2.Sobel(self.im, cv2.CV_64F, 0, 1, ksize=3)

		# calculate absolute magnitude of the gradient
		magnitude = np.sqrt(gx ** 2 + gy ** 2)

	    # set up gradient
		gradient = np.zeros((self.heigh, self.width, self.numorient))

		# calc initial gradient orientation and magnnitude
		mid = self.numorient / 2
		for y in range(self.heigh):
			for x in range(self.width):
				# normalize the gradient vector
				magnitude_norm = np.sqrt((gy[y, x] ** 2 + gx[y, x] ** 2) / magnitude[y, x] ** 2 if magnitude[y, x] > 0 else 0)
				
				orientation = 0
				if magnitude_norm[y, x] >= self.magnitude_threshold:
					angle = np.arctan2(gy[y, x], gx[y, x])
					orientation = math.floor(1 + angle / (np.pi / mid))
					gradient[y, x, orientation] += magnitude_norm

		self.integral_gradient = np.copy(gradient)
		# calc gradient integral image
		for y in xrange(1, self.heigh):
			for x in xrange(1, self.width):
				for ang in range(self.numorient):
					self.integral_gradient[y, x, ang] += gradient[y, x, ang] + \
														gradient[y - 1, x, ang] + \
														gradient[y, x - 1, ang] - \
														gradient[y - 1, x - 1, ang]

	"""
		Calculate HOG features for a given cell from integral gradient image
		returns:
			CoHOG features not-normalized
		params:
			cell: cell position
		"""	
	def _calc_cell_HOG(self, cell):
		features = np.zeros(self.numorient)
		for ang in range(self.numorient):
			features[ang] = self.integral_gradient[cell[0], cell[1], ang] + \
							self.integral_gradient[cell[2], cell[3], ang] - \
							self.integral_gradient[cell[2], cell[1], ang] - \
							self.integral_gradient[cell[0], cell[3], ang]
		return features

	def _calc_descriptor(self):
		# calculate cell descriptor
		h, w = self.heigh / self.pixels_per_cell[0], self.width / self.pixels_per_cell[1]
		
		descriptor = []

		# calculate block descriptor
		overlap = 0.5
		stepY = overlap * cells_per_block[0]
		stepX = overlap * cells_per_block[1]

		for y in xrange(0, h - cells_per_block[0], stepY):
			for x in xrange(0, w - cells_per_block[1], stepX):
				# L1-norm block normalization
				fd = []

				# calculate hog vector for each cell in block
				for i in range(cells_per_block[0]):
					for j in range(cells_per_block[1]):
						self._calc_cell_HOG((i, j, i + 1, j + 1))

	def compute(self, im, numorient = 9, pixels_per_cell = (8, 8), cells_per_block=(2, 2), magnitude_threshold = 0.0):
		# initialize parameters
		self.im = im
		self.numorient = numorient
		self.magnitude_threshold = magnitude_threshold
		self.pixels_per_cell = pixels_per_cell
		self.cells_per_block = cells_per_block

		# preparing image
		self.heigh, self.width = self.im.shape

		# calculate weighted gradient
		self._calc_weighted_gradient()