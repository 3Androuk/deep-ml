
import numpy as np

def dice_score(y_true, y_pred):
	tp = ((y_true == 1) & (y_pred == 1)).sum()
	fp = ((y_true == 0) & (y_pred == 1)).sum()
	fn = ((y_true == 1) & (y_pred == 0)).sum()

	dn = 2 * tp + fp + fn
	if dn == 0:
		return 0.0

	res = 2 * tp / dn
	return round(res, 3)
