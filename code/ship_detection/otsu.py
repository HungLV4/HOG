import numpy as np

def otsu(data, size_column, size_row):
	max_val = 100
	data = np.floor(data * max_val)
	binwidth = 1

	# Calculate the histogram of image
	hist, bin_edges = np.histogram(data, bins=np.arange(0, max_val + 2 * binwidth, binwidth))
	total = size_row * size_column

	current_max, threshold = 0.0, 0.0
	sumT, sumF, sumB = 0, 0, 0

	for i in range(max_val + 1):
		sumT += i * hist[i]
	
	weightB, weightF = 0, 0
	varBetween, meanB, meanF = 0.0, 0.0, 0.0
	for i in range(max_val + 1):
		weightB += hist[i]
		weightF = total - weightB
		if weightF == 0:
			break
		
		sumB += i * hist[i]
		sumF = sumT - sumB
		
		meanB = sumB / weightB
		meanF = sumF / weightF
		
		varBetween = weightB * weightF * ((meanB - meanF) ** 2)
		if varBetween > current_max:
			current_max = varBetween
			threshold = i

	return threshold



