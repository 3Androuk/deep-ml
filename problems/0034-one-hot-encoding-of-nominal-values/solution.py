import numpy as np

def to_categorical(x, n_col=None):

	output = []
	if n_col is None:
		n_col = max(x+1)
	for i in x:
		v = (np.arange(n_col) == i).astype(float)
		output.append(v)
	return np.array(output)