
import numpy as np

def confusion_matrix(data):
	data
	y_true, y_pred = map(np.array, zip(*data)) 
	tp = np.sum(((y_true == 1) & (y_pred == 1)))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	tn = np.sum((y_true == 0) & (y_pred == 0))
	fn = np.sum((y_true == 1) & (y_pred == 0))

	cm = [[tp, fn], [fp, tn]]
	return cm



