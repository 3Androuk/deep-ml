import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	mu = np.mean(data, axis=0)
	sig = np.std(data, axis=0)
	standardized_data = (data - mu)/sig
	
	ma = np.max(data, axis=0)
	mi = np.min(data, axis=0)
	normalized_data = (data - mi)/(ma - mi)
	return standardized_data, normalized_data