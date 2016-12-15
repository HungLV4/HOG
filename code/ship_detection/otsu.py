import numpy as np

def calc_otsu(data, size_column, size_row):
	max_val = 10000
	data = (np.floor(data * max_val)).astype('int')
	binwidth = 1

	# Calculate the histogram of image
	hist, bin_edges = np.histogram(data, bins=np.arange(0, max_val + 2 * binwidth, binwidth))

	# histogram equalization
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max() / cdf.max()
	cdf_m = np.ma.masked_equal(cdf, 0)
	cdf_m = (cdf_m - cdf_m.min()) * max_val / (cdf_m.max() - cdf_m.min())
	cdf = np.ma.filled(cdf_m, 0).astype('int')

	newData = cdf[data]
	hist, bin_edges = np.histogram(newData, bins=np.arange(0, max_val + 2 * binwidth, binwidth))

	# 
	total = size_row * size_column

	current_max, threshold = 0.0, 0
	sumT, sumF, sumB = 0.0, 0.0, 0.0

	for i in range(max_val + 1):
		sumT += i * hist[i]
	
	weightB, weightF = 0.0, 0.0
	varBetween, meanB, meanF = 0.0, 0.0, 0.0
	for i in range(max_val + 1):
		weightB += hist[i]
		if weightB == 0:
			continue

		weightF = total - weightB
		if weightF == 0:
			break
		
		sumB += i * hist[i]
		meanB = sumB / weightB

		sumF = sumT - sumB
		meanF = sumF / weightF
		
		varBetween = weightB * weightF * ((meanB - meanF) ** 2)
		if varBetween > current_max:
			current_max = varBetween
			threshold = i

	return float(threshold) / max_val