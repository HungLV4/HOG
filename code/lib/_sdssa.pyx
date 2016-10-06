# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from libc.math cimport sqrt

cimport numpy as cnp

cdef float texture_abnormal_pixel_calc(int[:, ::1] data,
				int column_ind, int row_ind,
				int size_column, int size_row,
				int size_window_column, int size_window_row,
				double[:, ::1] integral,
				double[:, ::1] integral_sqr) nogil:
	
	cdef float S1, S2
	cdef float rxd = 0.0
		
	cdef int tnrow = row_ind - size_window_row
	cdef int bnrow = row_ind + size_window_row
	
	cdef int lncol = column_ind - size_window_column
	cdef int rncol = column_ind + size_window_column
	
	if lncol < 0 or rncol >= size_column or tnrow < 0 or bnrow >= size_row:
		return rxd

	S1 = integral[bnrow, rncol] + integral[tnrow, lncol] - integral[bnrow, lncol] - integral[tnrow, rncol]
	S2 = integral_sqr[bnrow, rncol] + integral_sqr[tnrow, lncol] - integral_sqr[bnrow, lncol] - integral_sqr[tnrow, rncol]

	cdef int size_window_pixels = (bnrow - tnrow + 1) * (rncol - lncol + 1)
	if S1 > 0:
		rxd = sqrt(size_window_pixels * S2 - (S1 ** 2)) / S1

	return rxd

def texture_abnormal_calc(int[:, ::1] data,
				int size_column, int size_row,
				int size_window_column, int size_window_row,
				double[:, ::1] integral,
				double[:, ::1] integral_sqr,
				float[:, :] output):
	cdef int x, y

	""" Calculate the texture abnormality
	"""
	with nogil:
		for y in range(size_window_row, size_row - size_window_row - 1):
			for x in range(size_window_column, size_column - size_window_column - 1):
				output[y, x] = texture_abnormal_pixel_calc(data, x, y,
								size_column, size_row, 
								size_window_column, size_window_row, 
								integral, integral_sqr)

def integral_calc(int[:, ::1] data,
				int size_column, int size_row,
				double[:, :] integral,
				double[:, :] integral_sqr):
	cdef int x, y
	for y in range(size_row):
		for x in range(size_column):
			integral[y, x] = data[y, x] + \
							(integral[y - 1, x] if y - 1 >= 0 else 0) + \
							(integral[y, x - 1] if x - 1 >= 0 else 0) - \
							(integral[y - 1, x - 1] if (x - 1 >= 0 and y - 1 >= 0) else 0)

			integral_sqr[y, x] = data[y, x] ** 2 + \
							(integral_sqr[y - 1, x] if y - 1 >= 0 else 0) + \
							(integral_sqr[y, x - 1] if x - 1 >= 0 else 0) - \
							(integral_sqr[y - 1, x - 1] if (x - 1 >= 0 and y - 1 >= 0) else 0)
