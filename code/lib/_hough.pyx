# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from operator import itemgetter
from libc.math cimport sqrt, floor, atan2, sin, cos, M_PI, round

def ll_angle(unsigned char[:, ::1] img, 
			int size_columns, int size_rows,
			float NOTDEF):
	"""
	Calculating gradient magnitude and level-line angle
	Input:
		img: original 8-bit image
	Output:
		modgrad: gradient magnitude image
		modang: gradient angle image
	"""
	cdef int i, j
	
	cdef float quant = 2.0 # Bound to the quantization error on the gradient norm
	cdef float ang_th = 22.5 # Gradient angle tolerance in degrees

	# calculating gradient magnitude threshold
	cdef float prec = M_PI * ang_th / 180.0
	cdef float threshold_grad = quant / sin(prec)
	
	cdef float gx, gy, norm2, norm
	
	# initialize gradient magnitude and angle level-line data
	cdef np.ndarray[np.float32_t, ndim=2] modgrad = np.zeros((size_rows, size_columns), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=2] modang = np.zeros((size_rows, size_columns), dtype=np.float32)
	with nogil:
		for i in range(size_rows):
			for j in range(size_columns):
				if j == size_columns - 1 or i == size_rows - 1:
					modgrad[i, j] = NOTDEF
				else:
					# gradient x component
					gx = (img[i, j + 1] + img[i + 1, j + 1] - img[i, j] - img[i + 1, j]) / 2.0

					# gradient y component
					gy = (img[i + 1, j] + img[i + 1, j + 1] - img[i, j] - img[i, j + 1]) / 2.0

					norm2 = gx * gx + gy * gy
					# gradient norm
					norm = sqrt(norm2 / 4.0)
					if norm <= threshold_grad:
						modgrad[i, j] = NOTDEF
					else:
						modgrad[i, j] = norm
						modang[i, j] = atan2(gx, -gy)
	return modgrad, modang

def hough_lines(float[:, ::1] modgrad,
						int size_columns, int size_rows,
						float rho, float theta, int threshold_hough,
						double min_theta, double max_theta,
						int maxLines, float NOTDEF, int mode):
	cdef int i, j, n, r, idx
	cdef int base
	cdef float irho = 1 / rho
	cdef int numangle = int(round((max_theta - min_theta) / theta))
	cdef int numrho = int(round(((size_columns + size_rows) * 2 + 1) / rho))
	cdef double scale = 1. / (numrho + 2)

	cdef float rho_t, angle_t

	cdef list _sort_buf = []
	cdef list lines = []
	
	"""
	Transform to Hough space using weighted gradient
	Input:
		modgrad: gradient magnitude
	Ouput:
		lines: list of segment
		accum_vis (Optional): visualization of Hough space accumulator
	"""
	cdef np.ndarray[np.float32_t, ndim=1] accum = np.zeros((numangle + 2) * (numrho + 2), dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=1] tabSin = np.zeros(numangle, dtype=np.float32)
	cdef np.ndarray[np.float32_t, ndim=1] tabCos = np.zeros(numangle, dtype=np.float32)

	for n in range(numangle):
		tabSin[n] = float(sin(np.float64(min_theta + n * theta)) * irho)
		tabCos[n] = float(cos(np.float64(min_theta + n * theta)) * irho)

	""" stage 3.1. fill accumulator """
	with nogil:
		for i in range(size_rows):
			for j in range(size_columns):
				if modgrad[i, j] != NOTDEF:
					for n in range(numangle):
						r = int(round(j * tabCos[n] + i * tabSin[n]))
						r += (numrho - 1) / 2
						if mode == 0:
							accum[(n + 1) * (numrho + 2) + r + 1] += 1
						else:
							accum[(n + 1) * (numrho + 2) + r + 1] += modgrad[i, j]

	""" stage 3.2. find local maximums """
	for r in range(numrho):
		for n in range(numangle):
			base = (n + 1) * (numrho + 2) + r + 1
			if accum[base] > threshold_hough and \
				accum[base] > accum[base - 1] and \
				accum[base] >= accum[base + 1] and \
				accum[base] > accum[base - numrho - 2] and \
				accum[base] >= accum[base + numrho + 2]:
				_sort_buf.append((base, accum[base]))
	
	""" stage 3.3. sort the detected lines by accumulator value"""
	_sort_buf = sorted(_sort_buf, key=itemgetter(1))

	"""stage 3.4. store the first lines to the output buffer"""
	maxLines = maxLines if maxLines < len(_sort_buf) else len(_sort_buf)
	for i in range(maxLines):
		line = _sort_buf[len(_sort_buf) - 1 - i] 
		
		idx = line[0]
		n = int(floor(idx * scale) - 1)
		r = idx - (n + 1) * (numrho + 2) - 1
        
		rho_t = (r - (numrho - 1) * 0.5) * rho
		angle_t = float(min_theta) + n * theta
		lines.append((rho_t, angle_t))

	cdef np.ndarray[np.float32_t, ndim=2] accum_vis = np.zeros((numangle + 2, numrho + 2), dtype=np.float32)
	accum_vis = accum.reshape((numangle + 2, numrho + 2))
	
	return lines, accum_vis


