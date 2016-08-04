import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from joblib import load, dump
import tempfile
import os

def cell_hog(i, magnitude, orientation, 
            bin_width, 
            cell_columns, cell_rows, 
            size_columns, size_rows, 
            range_rows_start, range_rows_stop, 
            range_columns_start, range_columns_stop,
            number_of_cells_columns, number_of_cells_rows,
            orientation_histogram):
    """Calculation of the cell's HOG value
    Parameters:
        magnitude : ndarray
            The gradient magnitudes of the pixels.
        orientation : ndarray
            Lookup table for orientations.
        bin_width : float
            Orientation bin width.
        cell_columns : int
            Pixels per cell (rows).
        cell_rows : int
            Pixels per cell (columns).
        size_columns : int
            Number of columns.
        size_rows : int
            Number of rows.
        range_rows_start : int
            Start row of cell.
        range_rows_stop : int
            Stop row of cell.
        range_columns_start : int
            Start column of cell.
        range_columns_stop : int
            Stop column of cell
        number_of_cells_columns: int
            Number of cells (column)
        number_of_cells_rows: int
            Number of cells (row)
        orientation_histogram: ndarray
            Computed integral histogram
    """

    # isolate orientations in this range
    orientation_start = bin_width * (i + 1)
    orientation_end = bin_width * i
    
    x = cell_columns / 2
    y = cell_rows / 2
    yi = 0
    xi = 0
    while y < cell_rows * number_of_cells_rows:
        xi = 0
        x = cell_columns / 2

        while x < cell_columns * number_of_cells_columns:
            # calculate the integral
            total = 0.0
            for cell_row in range(range_rows_start, range_rows_stop):
                cell_row_index = y + cell_row
                if (cell_row_index < 0 or cell_row_index >= size_rows):
                    continue

                for cell_column in range(range_columns_start, range_columns_stop):
                    cell_column_index = x + cell_column
                    if (cell_column_index < 0 or cell_column_index >= size_columns
                            or orientation[cell_row_index, cell_column_index]
                            >= orientation_start
                            or orientation[cell_row_index, cell_column_index]
                            < orientation_end):
                        continue

                    total += magnitude[cell_row_index, cell_column_index]

            orientation_histogram[yi, xi, i] = total / (cell_rows * cell_columns)
            
            xi += 1
            x += cell_columns

        yi += 1
        y += cell_rows

def hog_histograms(gradient_columns, gradient_rows,
                   cell_columns, cell_rows,
                   size_columns, size_rows,
                   number_of_cells_columns, number_of_cells_rows,
                   number_of_orientations):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.
    Parameters:
        gradient_columns : ndarray
            First order image gradients (rows).
        gradient_rows : ndarray
            First order image gradients (columns).
        cell_columns : int
            Pixels per cell (rows).
        cell_rows : int
            Pixels per cell (columns).
        size_columns : int
            Number of columns.
        size_rows : int
            Number of rows.
        number_of_cells_columns : int
            Number of cells (rows).
        number_of_cells_rows : int
            Number of cells (columns).
        number_of_orientations : int
            Number of orientation bins.
    
    Return:
        orientation_histogram : ndarray
            The histogram array which is modified in place.
    """
    
    x0 = cell_columns / 2
    y0 = cell_rows / 2
    cc = cell_rows * number_of_cells_rows
    cr = cell_columns * number_of_cells_columns
    number_of_orientations_per_180 = 180. / number_of_orientations
    range_rows_stop = cell_rows/2
    range_rows_start = -range_rows_stop
    range_columns_stop = cell_columns/2
    range_columns_start = -range_columns_stop

    """ The first step to calculate the magnitude and orientation from
        gradients
    """
    magnitude = np.hypot(gradient_columns, gradient_rows)
    orientation = np.arctan2(gradient_rows, gradient_columns) * (180 / np.pi) % 180

    """ The second step is to calculate the integral histogram
    """

    # filepath for temp generated file used by parallel computation
    folder = tempfile.mkdtemp()
    magnitude_path = os.path.join(folder, 'mag')
    orientation_path = os.path.join(folder, 'ori')
    hist_path = os.path.join(folder, 'hist')

    # Pre-allocate a writeable shared memory map as a container for the results
    # orientation histogram of the parallel computation
    orientation_histogram = np.memmap(hist_path, dtype=np.float64,
                            shape=(number_of_cells_rows, number_of_cells_columns, number_of_orientations), 
                            mode='w+')
    # Dump the input images to disk to free the memory
    dump(magnitude, magnitude_path)
    dump(orientation, orientation_path)

    # Release the reference on the original in memory array and replace it
    # by a reference to the memmap array so that the garbage collector can
    # release the memory before forking. gc.collect() is internally called
    # in Parallel just before forking.
    magnitude = load(magnitude_path, mmap_mode='r')
    orientation = load(orientation_path, mmap_mode='r')

    # get the number of system cores
    num_cores = multiprocessing.cpu_count()
    
    # Fork the worker processes to perform motion vector computation concurrently
    Parallel(n_jobs=num_cores)(delayed(cell_hog)(i, magnitude, orientation, number_of_orientations_per_180, cell_columns, cell_rows, size_columns, size_rows, range_rows_start, range_rows_stop, range_columns_start, range_columns_stop, number_of_cells_columns, number_of_cells_rows, orientation_histogram) for i in range(number_of_orientations))

    # try:
    #     shutil.rmtree(folder)
    # except:
    #     print("Failed to delete: " + folder)

    return orientation_histogram