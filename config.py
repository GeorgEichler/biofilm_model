import numpy as np
import copy

class OneDConfig:
    """
    Config class for one dimensional thin film equation
    """
    
    def __init__(self, **kwargs):
        """
        The model has the following default parameters which can be overwritten

        Kwargs:
            L (float): Domain length [0, L]
            N (int): Number of grid points
            Q (float): Diffusion coefficient
            gamma (float): Surface tension
            h_max (float): maximal film height for the growth term
            g (float): growth coefficient
            a, b, c, d, k (float): Parameter for the binding potential
            h_init_type (str): Type of inital condition
        """
        # Default values
        self.params = {
            'L': 10, 'N': 1000, 'Q': 0.5, 'gamma': 0.1, 'h_max': 0.5, 'g': 0.1,
            'a': 0.1, 'b': np.pi/2, 'c': 1.0, 'd': 0.0, 'k': 2*np.pi
        }

        # Update parameters with possible user-provided arguments
        self.params.update(kwargs)

    def copy(self):
        """
        Create a deep copy of the configuration instance
        """
        return copy.deepcopy(self)
