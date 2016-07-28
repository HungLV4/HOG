import tempfile
import os

import numpy as np

from joblib import Parallel, delayed
from joblib import load, dump

import multiprocessing


""" Computes the Mean Absolute Difference (MAD)
"""
def evalMAD(mb1, mb2):
	height, width = mb1.shape
	cost = 0.0
	for i in range(height):
		for j in range(width):
			cost += abs(int(mb1[i, j]) - int(mb2[i, j]))
	return cost / (height * width)

def estTSS(curI, nextI, velX, velY, i, j, blockSize, stepSize, shiftSize, height, width):
	if i * shiftSize < blockSize or j * shiftSize < blockSize:
		velX[i, j] = 0
		velY[i, j] = 0
	else:
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
					
					# calculate cost at new position
					cost = evalMAD(origMb, refMb)
											
					if cost < minCost:
						minCost = cost
						newOrigX = refX
						newOrigY = refY
			
			_stepSize = _stepSize / 2
		
		velX[i, j] = newOrigX - i * shiftSize
		velY[i, j] = newOrigY - j * shiftSize
	
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
	
	# filepath for temp generated file used by parallel computation
	folder = tempfile.mkdtemp()
	curI_path = os.path.join(folder, 'curI')
	nextI_path = os.path.join(folder, 'nextI')
	velX_path = os.path.join(folder, 'velX')
	velY_path = os.path.join(folder, 'velY')

	# get pre-defined size
	height, width = curI.shape
	velSize = ((height - blockSize + shiftSize) / shiftSize, (width - blockSize + shiftSize) / shiftSize)
	
	# get the number of system cores
	num_cores = multiprocessing.cpu_count()

	# Pre-allocate a writeable shared memory map as a container for the results
	# motion vectors of the parallel computation
	velX = np.memmap(velX_path, dtype=np.int32, shape=velSize, mode='w+')
	velY = np.memmap(velY_path, dtype=np.int32, shape=velSize, mode='w+')

	# Dump the input images to disk to free the memory
	dump(curI, curI_path)
	dump(nextI, nextI_path)

	# Release the reference on the original in memory array and replace it
	# by a reference to the memmap array so that the garbage collector can
	# release the memory before forking. gc.collect() is internally called
	# in Parallel just before forking.
	curI = load(curI_path, mmap_mode='r')
	nextI = load(nextI_path, mmap_mode='r')

	# Fork the worker processes to perform motion vector computation concurrently
	Parallel(n_jobs=num_cores)(delayed(estTSS)(curI, nextI, velX, velY, i, j, blockSize, stepSize, shiftSize, height, width) for i in range(velSize[0]) for j in range(velSize[1]))

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