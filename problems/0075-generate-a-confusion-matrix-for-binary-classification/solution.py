
import numpy as np

def confusion_matrix(data):
	data_np = np.array(data)
	y_true = data_np[:, 0]
	y_pred = data_np[:, 1]
	
	tp = np.sum(((y_true == 1) & (y_pred == 1)))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	tn = np.sum((y_true == 0) & (y_pred == 0))
	fn = np.sum((y_true == 1) & (y_pred == 0))

	cm = [[tp, fn], [fp, tn]]
	return cm



