# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp

cdef float evalMAD(unsigned char[:, ::1] im, 
				unsigned char[:, ::1] next_im,
				int start_column_index, int stop_column_index,
				int start_row_index, int stop_row_index,
				int offset_column, int offset_row,
				int size_columns, int size_rows) nogil:
	""" Computes the Mean Absolute Difference (MAD)
	"""
	cdef int i, j
	cdef float cost

	cost = 0.0
	for i in range(start_row_index, stop_row_index):
		for j in range(start_column_index, stop_column_index):
			if im[i, j] > next_im[i, j]:
				cost += im[i, j] - next_im[i + offset_row, j + offset_column]
			else:
				cost += next_im[i + offset_row, j + offset_column] - im[i, j]
	
	return cost / (size_columns * size_rows)

cdef int estTSS(unsigned char[:, ::1] im, 
			unsigned char[:, ::1] next_im, 
			int column_index, int row_index, 
			int cell_columns, int cell_rows, int stepSize, 
			int size_columns, int size_rows) nogil:

	cdef int vy, vx, newX, newY, x, y, \
		start_column_index, stop_column_index, \
		start_row_index, stop_row_index, \
		offset_column, offset_row, \
		_offset_column, _offset_row, \
		min_offset_column, min_offset_row, \
		_stepSize
	
	cdef float cost, minCost, minY, minX

	start_column_index = column_index * cell_columns
	stop_column_index = (column_index + 1) * cell_columns
	start_row_index = row_index * cell_rows
	stop_row_index = (row_index + 1) * cell_rows

	offset_column = 0
	offset_row = 0

	minCost = evalMAD(im, next_im, 
		start_column_index, stop_column_index, 
		start_row_index, stop_row_index,
		offset_column, offset_row,
		cell_columns, cell_rows)

	_stepSize = stepSize
	while _stepSize >= 1:
		min_offset_column = 0
		min_offset_row = 0
		for x in range(-1, 2):
			for y in range(-1, 2):
				if x == 0 and y == 0:
					continue
				
				_offset_column = y * stepSize
				_offset_row = x * stepSize

				if start_column_index + offset_column + _offset_column < 0 or \
					stop_column_index + offset_column + _offset_column >= size_columns or \
					start_row_index + offset_row + _offset_row < 0 or \
					stop_row_index + offset_row + _offset_row < 0:
					continue

				cost = evalMAD(im, next_im, 
					start_column_index, stop_column_index,
					start_row_index, stop_row_index,
					offset_column + _offset_column, offset_row + _offset_row,
					cell_columns, cell_rows)

				if cost < minCost:
					minCost = cost
					min_offset_column = _offset_column
					min_offset_row = _offset_row

		offset_column += min_offset_column
		offset_row += min_offset_row
		_stepSize = _stepSize / 2

	return offset_row * 10 + offset_column

def motionEstTSS(unsigned char[:, ::1] im, 
				unsigned char[:, ::1] next_im,
				int size_columns, int size_rows,
				int cell_columns, int cell_rows, int stepSize,
				int number_of_cells_columns, int number_of_cells_rows,
				cnp.float64_t[:, ::1] vx,
				cnp.float64_t[:, ::1] vy):
	cdef int i, j
	cdef int v
	with nogil:
		for i in range(number_of_cells_rows):
			for j in range(number_of_cells_columns):
				v = estTSS(im, next_im, j, i, cell_columns, cell_rows, stepSize, number_of_cells_columns, number_of_cells_rows)
				vy[i, j] = v / 10
				vx[i, j] = v % 10