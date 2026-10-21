import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	tp = ((y_true == 1) & (y_pred == 1)).sum()
	fp = ((y_true == 0) & (y_pred == 1)).sum()
	fn = ((y_true == 1) & (y_pred == 0)).sum()

	dn_p = tp + fp
	dn_r = tp + fn

	if dn_p == 0:
		precision = 0
	else:
		precision = tp / dn_p

	if dn_r == 0:
		recall = 0
	else:
		recall = tp / dn_r
	
	dn = beta ** 2 * precision + recall
	if dn == 0:
		score = 0
	else:
		score = (1 + beta ** 2) * (precision * recall) / dn
	
	return np.round(score, decimals=3)
