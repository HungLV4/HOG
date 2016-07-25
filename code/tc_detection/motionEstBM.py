import numpy as np

def evalMAD(mb1, mb2):
	height, width = mb1.shape
	cost = 0.0
	for i in range(height):
		for j in range(width):
			cost += abs(int(mb1[i, j]) - int(mb2[i, j]))
	return cost / (height * width)

# Computes motion vectors using 3-step search method
# Input
#   cur : The image for which we want to find motion vectors
#   next : The reference image
# 	blockSize:
# 	stepSize:
# 	shiftSize:
# Ouput
#    motionVect : the motion vectors for each integral macroblock in imgP
def motionEst3SS(curI, nextI, blockSize, stepSize, shiftSize):
	if nextI.shape != curI.shape:
		return []
	
	height, width = curI.shape

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
			
			while stepSize >= 1:
				minCost = -1.0				
				for x in xrange(-1,2,1):
					for y in xrange(-1,2,1):
						# calculate cost at new position
						refX = x * stepSize + newOrigX
						refY = y * stepSize + newOrigY
						if refX < blockSize or refY < blockSize or refX + blockSize >= height or refY + blockSize >= width:
							continue
						
						refMb = nextI[refX - blockSize: refX + blockSize, refY - blockSize: refY + blockSize]
						cost = evalMAD(origMb, refMb)
						
						if minCost == -1.0:
							minCost = cost
							newOrigX = refX
							newOrigY = refY
						else:
							if cost < minCost:
								minCost = cost
								newOrigX = refX
								newOrigY = refY
				
				stepSize = stepSize / 2
			velX[i, j] = newOrigX
			velY[i, j] = newOrigY

	return velX, velY