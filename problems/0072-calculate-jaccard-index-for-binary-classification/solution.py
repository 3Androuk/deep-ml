
import numpy as np

def jaccard_index(y_true, y_pred):
	result = (
		np.sum((y_true == 1) & (y_pred == 1)) /
		np.sum((y_true == 1) | (y_pred == 1))
	)
	return round(result, 3)
