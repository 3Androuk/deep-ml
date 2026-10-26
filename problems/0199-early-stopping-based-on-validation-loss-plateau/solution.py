def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
	"""
	Determine at each epoch whether training should stop based on validation loss.
	
	Args:
		val_losses: List of validation losses at each epoch
		patience: Number of epochs to wait for improvement before stopping
		min_delta: Minimum change in validation loss to qualify as improvement
	
	Returns:
		List of booleans indicating whether to stop at each epoch
	"""
	best_epoch = 0
	current = 0
	n = len(val_losses)
	output = [False]

	for i in range(1,n):
		diff = val_losses[best_epoch] - val_losses[i]
		if round(diff,10) <= min_delta:
			current += 1
		else:
			current = 0
			best_epoch = i
		
		if current >= patience:
			output.append(True)
		else:
			output.append(False)
	
	return output


