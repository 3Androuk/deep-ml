import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	loss = 0
	n = len(y_true)
	for i, val in enumerate(y_true):
		x = X[i]
		loss += (val - np.transpose(x) @ w) ** 2
	loss /= n
	for j in w:
		loss += alpha * j**2
	return np.round(loss, decimals=4)
