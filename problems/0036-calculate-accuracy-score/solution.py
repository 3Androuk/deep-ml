import numpy as np

def accuracy_score(y_true, y_pred):
	corr = (y_true == y_pred).sum()
	accuracy = corr / len(y_true)
	return accuracy