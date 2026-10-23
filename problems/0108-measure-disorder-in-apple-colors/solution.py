import numpy as np

def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	n = len(apples)
	if n == 0:
		return 0

	classes, counts = np.unique(apples, return_counts=True)
	prb = counts / n
	gini = 1 - np.sum(prb ** 2)
	return gini