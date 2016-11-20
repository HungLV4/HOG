# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from libc.math cimport sqrt, fabs

cimport numpy as cnp

def generate_fake_hs(int[:, ::1] data,
				int size_column, int size_row,
				int ws,
				int[:, :, :] output):
	cdef int x, y, i, j
	with nogil:
		for x in range(size_row):
			for y in range(size_column):
				for i in range(- ws / 2, ws / 2 + 1):
					for j in range(- ws / 2, ws / 2 + 1):
						if x + i >= 0 and x + i < size_row and y + j >= 0 and y + j < size_column:
							output[(i + ws / 2) * ws + (j + ws / 2), x, y] = data[x + i, y + j]