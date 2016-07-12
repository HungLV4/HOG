import cv2
import numpy as np
import itertools
import math

class HOGDetector(object):
	"""docstring for HOGDetector"""
	def __init__(self):
		super(HOGDetector, self).__init__()
		
	def _calc_weighted_gradient(self):
		"""
		Calculate integral gradient of image (w, h, numorient)
	          
	    returns 
	        Integral gradient image
	 
	    params 
	        im: original image
	        numorient: number of orientation bins, default is 9 (-4..4)
	    """
	    
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

	
	def _calc_hog_descriptor(self):
		for y in xrange(0, pixels_per_cell[0], self.heigh):
			for x in xrange(0, pixels_per_cell[1], self.width):
				fd = self._calc_cell_HOG(cell)

	def _block_normalization(self, gradient, pixels_per_cell = (8, 8), cells_per_block=(2, 2)):
		return gradient

	def _calc_cell_HOG(self, cell):
		"""
		Calculate HOG features for a given cell from integral gradient image
		returns:
			CoHOG features not-normalized
		params:
			cell: cell position
		"""	
		features = np.zeros(self.numorient)
		for ang in range(self.numorient):
			features[ang] = self.integral_gradient[cell[1], cell[0], ang] + \
							self.integral_gradient[cell[3], cell[2], ang] - \
							self.integral_gradient[cell[1], cell[2], ang] - \
							self.integral_gradient[cell[3], cell[0], ang]
		return features

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

		# calculate features
		for i in range(self.heigh):
			for j in range(self.width):
				# self._calc_cell_HOG((i, j, i + pixels_per_cell[0], j + pixels_per_cell[1]))
				pass

		self._block_normalization()
