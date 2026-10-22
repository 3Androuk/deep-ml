
import numpy as np

def rmse(y_true, y_pred):
	n = y_true.size
	ssr = np.sum((y_true - y_pred) ** 2)
	rmse_res = np.sqrt(ssr / n)
	return np.round(rmse_res,3)
