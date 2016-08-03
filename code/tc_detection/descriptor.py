import numpy as np
from histogram import *
from numpy import linalg as LA

""" L2-Normalization for a list
	Params:
		x: list of value
"""
def l2_normalization(x):
	l2_norm  = LA.norm(x)
	if l2_norm > 0:
		for i in range(len(x)):
			x[i] = x[i] / l2_norm
	return x

""" Calculate the descriptor of Circulate Histogram of Oriented AMV/Gradient
"""
def calcCHOGDescriptor(velX, velY, num_orient, threshold, radius):
	pass

def calcHOGDescriptor(velX, velY, orientations, ppc, cpb):
	""" Calculate the descriptor of Histogram of Oriented AMV/Gradient
		Params:
			velX, velY: (M, N) ndarray
				Input AMV/Gradients image of x-axis and y-axis resp
			num_orient: int
				Number of orientation bins.
			ppc: (int, int)
				Pixels per cell
			cpb: (int, int)
				Cells per block
	"""

	sy, sx = velX.shape
	cx, cy = ppc
	bx, by = cpb

	"""
    The first stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

	n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

	# compute the orientation integral images
	integral_hist = calcIHO(velX, velY, num_orient, amv_threshold)

	"""
    The second stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

	n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = integral_hist[y: y + by, x: x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / np.sqrt(block.sum() ** 2 + eps)

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    normalised_blocks = normalised_blocks.ravel()

	return normalised_blocks