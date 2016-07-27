from joblib import Parallel, delayed
import multiprocessing
import numpy as np

def absDiff(mb1, mb2):
	height, width = mb1.shape
	cost = 0
	for i in range(height):
		for j in range(width):
			cost += abs(int(mb1[i, j]) - int(mb2[i, j]))
	return cost

if __name__ == '__main__':
	# what are your inputs, and what operation do you want to 
	# perform on each input. For example...
	mb1 = np.zeros((800, 800))
	mb2 = np.zeros((800, 800))

	num_cores = multiprocessing.cpu_count()
	
	for k in range(10):
		print absDiff(mb1, mb2)
		

	with Parallel(n_jobs=num_cores) as parallel:
		cost = parallel(delayed(absDiff)(mb1, mb2) for k in range(10))
		print cost