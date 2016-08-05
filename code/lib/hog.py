import numpy as np
from _hoghistogram import hog_histograms

def calcCHOGDescriptor(gx, gy, num_orient, threshold, radius):
    """ Calculate the descriptor of Circulate Histogram of Oriented AMV/Gradient
    """
    pass

def calcHOGDescriptor(gx, gy, orientations, pixels_per_cell, cells_per_block):
    """ Calculate the descriptor of Histogram of Oriented AMV/Gradient
        Params:
            velX, velY: (M, N) ndarray
                Input AMV/Gradients image of x-axis and y-axis resp
            num_orient: int
                Number of orientation bins.
            pixels_per_cell: (int, int)
                Pixels per cell
            cells_per_block: (int, int)
                Cells per block
    """
    sy, sx = gx.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

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
    
    # number of cells in x
    n_cellsx = int(np.floor(sx // cx))

    # number of cells in y 
    n_cellsy = int(np.floor(sy // cy))

    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    hog_histograms(gx, gy, cx, cy, sx, sy, n_cellsx, n_cellsy, orientations, orientation_histogram)
    

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

    print n_blocksy, n_blocksx, by, bx, orientations
    normalised_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y: y + by, x: x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / np.sqrt(block.sum() ** 2 + eps)

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    normalised_blocks = normalised_blocks.ravel()

    return normalised_blocks