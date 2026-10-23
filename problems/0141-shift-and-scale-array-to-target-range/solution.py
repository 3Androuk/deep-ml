import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    a = np.min(values)
    b = np.max(values)

    transform = (d - c)/(b - a)
    new_values = c + transform * (values - a)
    return new_values
    