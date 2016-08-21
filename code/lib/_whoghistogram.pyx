# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from libc.math cimport sqrt, floor, atan2, cos, M_PI

cimport numpy as cnp

"""
 *  Orientation coding
 *     e.g., #bins = 8
 *           5 6 7
 *            \|/
 *          4-- --0       +--->x
 *            /|\         |
 *           3 2 1        v
 *                        y
"""
cdef void orientation_code(double[:, ::1] gx, double[:, ::1] gy,
                            int column_ind, int row_ind, 
                            int nbin,
                            double ztol,
                            double[:, :, :] hist) nogil:
    
    cdef float M_PI = 3.141592
    cdef double radian2bin = nbin/(2 * M_PI)
    
    cdef double theta
    cdef double mag, ind, wei
    cdef double dindex, index

    cdef double x = gx[row_ind, column_ind]
    cdef double y = gy[row_ind, column_ind]

    mag = sqrt(x * x + y * y)
    if mag < ztol:
        hist[row_ind, column_ind, 0] = -1
        hist[row_ind, column_ind, 1] = -1
        return
    
    hist[row_ind, column_ind, 0] = mag

    theta = atan2(y, x)
    if theta < 0:
        theta += 2 * M_PI;
    
    """ Mode 1: store absolute angle value
    """
    hist[row_ind, column_ind, 1] = theta
    
    """ Mode 2: store bin angle value
    """
    # dindex = theta * radian2bin
    # index = floor(dindex)
    # if index == nbin:
    #     hist[row_ind, column_ind, 1] = 0
    #     hist[row_ind, column_ind, 2] = 0
    # else:
    #     hist[row_ind, column_ind, 1] = index
    #     hist[row_ind, column_ind, 2] = 1 + index - dindex

cdef void calc_glac(double[:, :, :] data, 
                    int column_ind, int row_ind,
                    int nbin,
                    int size_columns, int size_rows,
                    int* rvecs, int nr,
                    double[:, :] ouput) nogil:
    
    cdef int i
    cdef int dx, dy
    cdef double mag2, theta2, theta

    cdef double mag1 = data[row_ind, column_ind, 0]
    if mag1 < 0:
        return

    """ Mode 1: data store absolute angle value
    """
    cdef double theta1 = data[row_ind, column_ind, 1]
    
    for i in range(nr):
        dx = column_ind + rvecs[i * 2]
        dy = row_ind + rvecs[i * 2 + 1]

        if dy >= 0 and dy < size_rows and dx >= 0 and dx < size_columns:
            mag2 = data[dy, dx, 0]
            if mag2 < 0:
                continue

            theta2 = data[dy, dx, 1]

            # mag = mag1 if mag1 < mag2 else mag2 
            
            theta = theta2 - theta1 if theta2 > theta1 else theta1 - theta2
            wei = cos(theta)
            if wei < 0:
                wei = -wei

            ouput[row_ind, column_ind] += mag1 * wei
    
    ouput[row_ind, column_ind] /= nr            
    
    """ Mode 2: data store bin angle value
    """
    # cdef int ind1 = int(data[row_ind, column_ind, 1])
    # cdef int ind1n = (ind1 + 1) % nbin

    # cdef double wei1 = data[row_ind, column_ind, 2]
    # cdef double wei1n = 1 - wei1

def whog_histograms(double[:, ::1] gx, double[:, ::1] gy,
                   int nbin, int size_columns, int size_rows,
                   double ztol,
                   double[:, :, :] orientation_histogram,
                   double[:, :] glac):
    cdef int i, j
    cdef int* rvecs = [0, 1, 1, 1, 1, 0, 1, -1, 1, -2, 0, 2, 1, 2, 2, 2, 2, 1, 2, 0, 2, -1, 2, -2]
    cdef int nr = 12
    
    with nogil:
        for i in range(size_rows):
            for j in range(size_columns):
                    orientation_code(gx, gy, j, i, nbin, ztol, orientation_histogram)

    with nogil:
        for i in range(size_rows):
            for j in range(size_columns):
                    calc_glac(orientation_histogram, j, i, nbin, size_columns, size_rows, rvecs, nr, glac)
    