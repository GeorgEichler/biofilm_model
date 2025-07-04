import numpy as np
from scipy.sparse import diags

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
            'a': 0.1, 'b': np.py/2, 'c': 1.0, 'd': 0.0, 'k': 2*np.pi,
            'h_init_type': 'gaussian' 
        }

        # Update parameters with possible user-provided arguments
        self.params.update(kwargs)

        self._setup_grid_and_operators()
        self._set_initial_condition()

    def _setup_grid_and_operators(self):
        """
        Define computational grid and finite difference matrices
        """
        N = self.params['N']
        L = self.params['L']

        self.dx = L / N
        self.x = (np.arange(1, N + 1) - 0.5) * self.dx # cell centered grid

        # First derivative with periodic boundary conditions
        D = diags([-1, 0, 1], [-1, 0, 1], shape=(N, N)).toarray() / (2 * self.dx)
        D[0, :] = 0 
        D[0, 1] = 1 /(2 * self.dx)
        D[0, -1] = -1 /(2 * self.dx)

        D[-1, :] = 0
        D[-1, 0] = 1 / (2 *self.dx)
        D[-1, -2] = -1 / (2 *self.dx)

        # Second derivative with periodic boundary conditions
        main_diag = -2.0 * np.ones(N)
        off_diag = np.ones(N - 1)
        Lap = diags([off_diag, main_diag, off_diag], [-1, 0 , 1]).toarray() /self.dx**2
        Lap[0, -1] = 1 / self.dx**2
        Lap[-1, 0] = 1 / self.dx**2
        self.Lap = Lap

    def _set_initial_condition(self):
        """Initial height profile of the thin film"""
        L = self.params['L']
        init_type = self.params['h_init_type']

        if init_type == 'gaussian':
            self.h_init = 0.5 + 5 * np.exp(-(self.x - L/2)**2 / 0.1)
        elif init_type == 'constant':
            self.h_init = np.ones_like(self.x)
        else:
            raise ValueError(f"Unknown h_init_type:{init_type}")
