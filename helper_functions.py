import numpy as np
from scipy.signal import find_peaks

def find_first_k_minima(k_minima, f, range = [0,10], num_points = 1000):
    """
    Find the first k minima of a function f(x)

    Args:
        k_minima (int): Number of minima to find
        f (func): Function to find minima of
        range (array): The search range
        num_points (int): Number of points for grid search

    Returns:
        x_minima (np.ndarray): values of minima
        f_minima (np.ndarray): corresponding function values to minima
    """

    # Create dense grid
    x_values = np.linspace(range[0], range[1], num_points)
    f_values = f(x_values)

    # find peaks of -f which are the minima
    indices, _ = find_peaks(-f_values, prominence=1e-4)

    if len(indices) < k_minima:
        num_minima = len(indices)
        print(f"Warning: Found only {num_minima} minima instead of {k_minima}.")
        print("Consider increasing the range or the grid number.")
    else:
        num_minima = k_minima

    first_k_indices = indices[:num_minima]
    x_minima = x_values[first_k_indices]
    f_minima = f_values[first_k_indices]

    return x_minima, f_minima