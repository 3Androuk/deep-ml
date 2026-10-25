def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    def delta_f(x):
      return 4 * x ** 3 - 9 * x ** 2

    alpha = 0.15
    x_old = start_x
    tolerance = 1e-2
    max_iter = 1000000

    for i in range(max_iter):
      x_new = x_old - alpha * delta_f(x_old)

      if abs(x_new - x_old) < 1e-2:
        return x_new
      
      x_old = x_new
    return x_new


    