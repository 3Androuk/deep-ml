import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed is not None:
		np.random.seed(seed)
	
	n = X.shape[0]
	idx = np.random.permutation(n)

	X_shuffled = X[idx]
	y_shuffled = y[idx]
	
	return X_shuffled, y_shuffled