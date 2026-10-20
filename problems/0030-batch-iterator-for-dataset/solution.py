import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	iter = int(np.ceil(len(X) / batch_size))
	output = []
	if y is not None:
		for i in range(iter):
			start = i * batch_size
			end = start + batch_size
			output.append([X[start:end], y[start:end]])
	else:
		for i in range(iter):
			start = i * batch_size
			end = start + batch_size
			output.append(X[start:end])
	return output
