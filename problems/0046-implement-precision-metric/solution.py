import numpy as np
def precision(y_true, y_pred):

	tp = ((y_true == 1) & (y_pred == 1)).sum()
	fp = ((y_true == 0) & (y_pred == 1)).sum()

	dn = tp + fp

	if dn == 0:
		precision = 0
	else:
		precision = tp / (dn)

	return precision
