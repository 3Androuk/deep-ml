def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(b) != len(a[0]):
		return -1
	output = [0] * len(a)
	for i, row in enumerate(a):
		for j, val in enumerate(row):
			output[i] += val * b[j]
	return output