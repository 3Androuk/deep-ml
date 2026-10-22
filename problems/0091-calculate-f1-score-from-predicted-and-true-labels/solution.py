import numpy as np

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))

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
	
	dn = precision + recall
	if dn == 0:
		f1 = 0
	else:
		f1 = 2 * precision * recall / dn
	
	
	return round(f1,3)