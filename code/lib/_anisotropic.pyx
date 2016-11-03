# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from libc.math cimport cos, fabs

cimport numpy as np

def calc_anisotropic_inhibterm(float[:, ::1] x,
						double [:, ::1] w,
						float [:, ::1] o,
						int size_col, int size_row,
						int k_size_col, int k_size_row,
						int mode):
	"""
	Calculate the anisotropic suppression surround term
	Input:
		x: gradient madnitude
		w: kernel
		o: gradient orientation
		mode = 0: isotropic suppresion
			 = 1: anisotropic suppression
			 = 2: invert anisotropic suppression
			 = 3: custom
	"""
	
	# initialize response signal
	cdef np.ndarray[np.float32_t, ndim=2] response = np.zeros((size_row, size_col), dtype=np.float32)
	cdef int kCenterX = k_size_col / 2
	cdef int kCenterY = k_size_row / 2

	cdef int i, j, m, n, mm, nn, ii, jj
	with nogil:
		for i in range(size_row):
			for j in range(size_col):
				# kernel rows
				for m in range(k_size_row):
					# row index of flipped kernel
					mm = k_size_row - 1 - m
					# kernel cols
					for n in range(k_size_col):
						# column index of flipped kernel
						nn = k_size_col - 1 - n

						# index of input signal, used for checking boundary
						ii = i + (m - kCenterY)
						jj = j + (n - kCenterX)

						# ignore input samples which are out of bound
						if ii >= 0 and ii < size_row and jj >= 0 and jj < size_col:
							if mode == 1:
								response[i, j] += x[ii, jj] * w[mm, nn] * fabs(cos(o[i, j] - o[ii, jj]))
							elif mode == 0:
								response[i, j] += x[ii, jj] * w[mm, nn]
							elif mode == 2:
								response[i, j] += x[ii, jj] * w[mm, nn] * (1 - fabs(cos(o[i, j] - o[ii, jj])))
							elif mode == 3:
								response[i, j] += x[ii, jj] * w[mm, nn] * (x[ii, jj] - x[i, j]) * (1 - fabs(cos(o[i, j] - o[ii, jj])))
	return response