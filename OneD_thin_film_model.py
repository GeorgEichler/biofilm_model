import numpy as np
import os
from abc import ABC, abstractmethod
from helper_functions import find_first_k_minima

class OneD_Base_Model(ABC):
    """
    Abstract base class for 1D thin-film model
    handling physical paramters, grid and initial conditions
    """

    def __init__(self, **kwargs):
        """
        Kwargs:
            L (float): Domain length [0, L]
            N (int): Number of grid points
            h_max (float): maximal film height for the growth term
            g (float): coefficient of logistic growth, qoutient of growth and diffusion coefficient
            a, b, c, d, e, k (float): parameter for the binding potential
            h_init_type (str): Type of inital condition
            amplitude (float): amplitude of the binding potential 
        """
        # Default values
        self.params = {
            'L': 100, 'N': 1024, 'gamma': 1, 'g': 10**(-4), 'ha': 0.8, 'hf': 0.1, 'h_max': 5, 'Q': 1.0, 'epsilon': 1.0,
            'a': 0.5, 'b': np.pi, 'c': 1.0, 'd': 10, 'e': 0.01, 'k': 2*np.pi,
            'amplitude': 1.0, 'var': 10
        }
        self.params.update(kwargs)

        p = self.params
        self.dx = p['L'] / p['N'] # spacial discretisation
        self.x = (np.arange(1, p['N'] + 1) - 0.5) * self.dx

        self._setup_numerical_operators()

        # Calculate equilibrium heights h0 and h1
        minima, _ = find_first_k_minima(3, self.f)
        self.h0 = minima[0]
        self.h1 = minima[1]
        self.h2 = minima[2]
        self.ha = p['ha'] # activation point
        self.hf = p['hf']

    @abstractmethod
    def _setup_numerical_operators(self):
        """
        Abstract method to define necessary operators like finite difference matrics
        or wavenumbers for spectral method
        """
        pass

    @abstractmethod
    def solve(self, h0, T, t_eval = None):
        """
        Abstract method to solve the PDE
        """
        pass

    def setup_initial_conditions(self, init_type):
        p = self.params
        L, amplitude, var = p['L'], p['amplitude'], p['var']
        
        if init_type == 'gaussian':
            h_init = (self.h0 + 0.01) + amplitude * np.exp(-(self.x - L/2)**2/var)
        elif init_type == 'double gaussian':
            x1 = 0.3 * L; x2 = 0.7 * L
            h_init = (self.h0 + 0.01) + amplitude * np.exp(-(self.x - x1)**2/var) + amplitude * np.exp(-(self.x - x2)**2/var)
        elif init_type == 'constant':
            h_init = np.full_like(self.x, self.h0 + 0.5)
        elif init_type == 'cap':
            h_init = np.maximum(self.h0 + 0.01, amplitude - 1/var * (self.x - L/2)**2)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        
        return h_init

    # Define binding energies and corresponding disjoint pressures
    def f(self, h):
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; e = p['e']; k = p['k']
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-h/e)

    def Pi(self, h):
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; e = p['e']; k = p['k']
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/e*np.exp(-h/e)
    
    def growth_term(self, h):
        p = self.params
        #growth = p['g'] * (h - self.ha) * (1 - h/p['h_max']) * (1 - np.exp( (self.h0 - h) ))

        growth = p['g'] * (h - self.ha) * (1 - h/p['h_max']) * (1 - np.exp( p['hf'] - h ))

        #growth = np.maximum(growth, 0) # alternative growth term with truncated growth

        return growth
    
    def save_profile_values(self, t: np.ndarray, H: np.ndarray, filename: str):
        """
        Save time points and height profiles to a npz file

        Parameters:
        t: 1D array of time values
        H: 2D array of height values
        filename: file path (should end in .npz)
        """

        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok = True)

        np.savez_compressed(filename, t=t, H=H)
        print(f"Saved profiles to {filename}")

    @staticmethod
    def load_profiles(filename: str):
        """
        Load time points and height profiles from npz file

        Returns:
        t: 1D array of time values
        H: 2D array of height values
        """
        data = np.load(filename)
        t = data['t']
        H = data['H']
        print(f"Loaded profile from {filename}")
        return t, H