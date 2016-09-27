import numpy as np
from scipy import signal
import cv2

modgrad = np.array([[1,2,3,], [3,4,5], [6,7,8]])
rows = cols = 3

sigma = 1.5
kernel_size = 3

# Get 1d Gaussian kernel
k1d_1sig = cv2.getGaussianKernel(kernel_size, sigma)
k1d_1sig = k1d_1sig.reshape(kernel_size)

# Generate 2d kernel (no lib support 2d kernel generation)
k2d_1sig = np.zeros((kernel_size, kernel_size))
for i in range(kernel_size):
	k2d_1sig[i] = k1d_1sig[i] * k1d_1sig

out2d = signal.convolve2d(modgrad, k2d_1sig, mode='same')
print out2d

# Calculate the covolution of edge with kernel 2d
# by convolve it in each direction
temp = None
for i in range(rows):
	if temp == None:
		temp = np.convolve(modgrad[i], k1d_1sig, mode='same')
	else:
		temp = np.row_stack((temp, np.convolve(modgrad[i], k1d_1sig, mode='same')))
# print temp

temp2 = None
for i in range(cols):
	# print np.convolve(temp[:, i], k1d_1sig, mode='full')
	if temp2 == None:
		temp2 = np.convolve(temp[:, i], k1d_1sig, mode='same')
	else:
		temp2 = np.column_stack((temp2, np.convolve(temp[:, i], k1d_1sig, mode='same')))

print ""
print temp2