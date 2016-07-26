import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import operator

""" Computes the Mean Absolute Difference (MAD)
"""
def evalMAD(mb1, mb2):
	height, width = mb1.shape
	cost = 0.0
	for i in range(height):
		for j in range(width):
			cost += abs(int(mb1[i, j]) - int(mb2[i, j]))
	return cost / (height * width)
	
""" Computes motion vectors using 3-step search method
Input:
	curI: The image for which we want to find motion vectors
	nextI: The reference image
	blockSize:
 	stepSize:
	shiftSize:
Ouput:
    velX, velY : the motion vectors for each direction
"""
def motionEstTSS(curI, nextI, blockSize, stepSize, shiftSize):
	# check if two images have the same size
	if nextI.shape != curI.shape:
		print "Two images do not have the same size"
		return [], []
	
	height, width = curI.shape

	# get the number of system cores
	num_cores = multiprocessing.cpu_count()

	# initialize motion vector
	velSize = ((height - blockSize + shiftSize) / shiftSize, (width - blockSize + shiftSize) / shiftSize)
	velX = np.zeros(velSize, dtype=np.int32)
	velY = np.zeros(velSize, dtype=np.int32)
	
	for i in xrange(velSize[0]):
		for j in xrange(velSize[1]):
			if i * shiftSize < blockSize or j * shiftSize < blockSize:
				continue
			
			# original block
			newOrigX = i * shiftSize
			newOrigY = j * shiftSize
			
			origMb = curI[newOrigX - blockSize : newOrigX + blockSize, 
						newOrigY - blockSize: newOrigY + blockSize]
			
			_stepSize = stepSize
			while _stepSize >= 1:
				# calculate cost at same position and initiate as minCost
				refMb = nextI[newOrigX - blockSize : newOrigX + blockSize, 
							newOrigY - blockSize: newOrigY + blockSize]
				minCost = evalMAD(origMb, refMb)

				listRef = []
				listXY = []
				for x in xrange(-1,2,1):
					for y in xrange(-1,2,1):
						if x == 0 and y == 0:
							continue

						# calculate ref position
						refX = x * _stepSize + newOrigX
						refY = y * _stepSize + newOrigY
						if refX < blockSize or refY < blockSize or refX + blockSize >= height or refY + blockSize >= width:
							continue
						
						# get the ref block
						refMb = nextI[refX - blockSize: refX + blockSize, refY - blockSize: refY + blockSize]
						
						# add to listRef for later calculate cost
						listRef.append(refMb)
						listXY.append((refX, refY))
				
				with Parallel(n_jobs=num_cores) as parallel:
					cost = parallel(delayed(evalMAD)(origMb, mb) for mb in listRef)

				min_index, min_cost = min(enumerate(cost), key=operator.itemgetter(1))
				if min_cost < minCost:
					newOrigX = (listXY[min_index])[0]
					newOrigY = (listXY[min_index])[0]
				
				_stepSize = _stepSize / 2
			
			velX[i, j] = newOrigX - i * shiftSize
			velY[i, j] = newOrigY - j * shiftSize

	return velX, velY

""" Computes motion vectors using 3-step search method
Input:
	curI: The image for which we want to find motion vectors
	nextI: The reference image
	blockSize:
	stepSize:
Ouput:
    velX, velY : the motion vectors for each direction
"""
def motionEstARPS(curI, nextI, blockSize, stepSize):
	# check if two images have the same size
	if nextI.shape != curI.shape:
		print "Two images do not have the same size"
		return [], []
	
	height, width = curI.shape

	# initialize motion vector
	velSize = ((height - blockSize + shiftSize) / shiftSize, (width - blockSize + shiftSize) / shiftSize)
	velX = np.zeros(velSize, dtype=np.int32)
	velY = np.zeros(velSize, dtype=np.int32)

	return velX, velY