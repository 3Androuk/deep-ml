def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	loss = 0
	if not isinstance(y_true, list):
		err = abs(y_true - y_pred)
		if err <= delta:
			loss += 0.5 * err ** 2
		else:
			loss += delta * (err - delta/2)
	else:
		n = len(y_true)
		for i, val in enumerate(y_true):
			err = abs(val - y_pred[i])
			if err <= delta:
				loss += 0.5 * err ** 2
			else:
				loss += delta * (err - delta/2)
		loss /= n
	return loss
