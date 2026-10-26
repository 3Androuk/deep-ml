import numpy as np

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.

	def f(i):
		return 10 * np.sin(2 * np.pi * i / 10)

	x = np.linspace(1, 10, 10)


	daily_fluctuation = f(x)
	y = consumption_data - daily_fluctuation

	X = np.vstack([x, np.ones_like(x)]).T

	w = np.linalg.inv(X.T @ X) @ X.T @ y

	pred = w[0] * 15 + w[1]
	final = 1.05 * pred
	
	return np.ceil(final)

