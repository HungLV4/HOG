import cv2
import numpy as np
import itertools

class CoHOGDetector(object):
	"""docstring for CoHOGDetector"""
	def __init__(self, im, numorient = 9, magnitude_threshold = 0.0):
		super(CoHOGDetector, self).__init__()
		self.im = im
		self.numorient = numorient
		self.magnitude_threshold = magnitude_threshold
	
	def _init(self):
		self.width, self.heigh, depth = self.im.shape
		if depth == 3:
			self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
		
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

	    # set up gradient count
	    self.im_gradient = np.zeros((self.heigh, self.width, self.numorient))

		# calc initial gradient orientation and magnnitude
		mid = self.numorient / 2
		for y in range(heigh - 1):
			for x in range(width - 1):
				magnitude = np.sqrt(gy[y, x] ** 2 + gx[y, x] ** 2)
				orientation = 0
				if magnitude >= self.magnitude_threshold:
					angle = np.arctan2(gy[y, x], gx[y, x])
					orientation = math.floor(1 + angle / (np.pi / mid))					
				
				self.im_gradient[y, x, orientation] += magnitude

	def _calc_cell_CoHOG(self, offsets):
		"""
		Calculate CoHOG features for a given cell from integral gradient image
		returns:
			CoHOG features not-normalized
		params:
			gradient: integral gradient image
			cell_size: cell position
		"""	
		features = np.zeros((self.numorient, self.numorient))
		for y in xrange(cell[1], cell[3]):
			for x in xrange(cell[0], cell[2]):
				# looping through neighbor pixels by offsets
				for offset in offsets:
					neigh_y = y + offset[1]
					neigh_x = x + offset[0]
					
					# check if neigborhood position is inside image
					if neigh_y < 0 or neigh_y > self.heigh - 1 or neigh_x < 0 or neigh_x > self.width - 1:
						continue
					
					# find co-occurence orientation
					for i, j in itertools.product(range(self.numorient), range(self.numorient)):
						features[i, j] += self.im_gradient[y, x, i] + self.im_gradient[neigh_y, neigh_x, j]
	    return features

	def compute(self, cell_size):
		pass