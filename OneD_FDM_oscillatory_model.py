import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import figure_handler as fh
import time

class OneD_Thin_Film_Model:
    """
    A class to set up a one dimensional thin-film equation model
    """

    def __init__(self, **kwargs):
        """
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
            'L': 10, 'N': 1000, 'Q': 0.5, 'gamma': 0.1, 'h_max': 5, 'g': 0.1,
            'a': 0.1, 'b': np.pi/2, 'c': 1.0, 'd': 0.0, 'k': 2*np.pi
        }

        # Update parameters with possible user-provided arguments
        self.params.update(kwargs)

        self._setup_grid_and_operators()

    def _setup_grid_and_operators(self):
        N = self.params['N']
        L = self.params['L']

        self.dx = L / N
        self.x = (np.arange(1, N + 1) - 0.5) * self.dx

        # First derivative with periodic boundary conditions
        D = diags(diagonals=[-1,1], offsets=[-1,1], shape=(N, N), format= 'lil')
        D[0, -1] = -1
        D[-1, 0] = 1

        self.D = (D / (2 * self.dx)).asformat('csr')

        # Second derivative with periodic boundary conditions
        Laplacian = diags(diagonals=[1,-2,1], offsets=[-1,0,1], shape = (N, N), format = 'lil')
        Laplacian[0, -1] = 1
        Laplacian[-1, 0] = 1

        self.Laplacian = (Laplacian / (self.dx**2)).asformat('csr')

    def setup_initial_conditions(self, init_type):
        L = self.params['L']

        if init_type == 'gaussian':
            h_init = 0.5 + 5 * np.exp(-(self.x - L/2)**2/0.1)
        elif init_type == 'constant':
            h_init = np.ones_like(self.x)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        
        return h_init

    def g1(self, h):
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; k = p['k']
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-h/(2*c))

    def g2(self, h):
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; k = p['k']
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-2*h/c)

    def Pi1(self, h):
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; k = p['k']
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))

    def Pi2(self, h):
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; k = p['k']
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + (2*d)/(c)*np.exp(-(2*h)/(c))
    
    def free_energy(self, h):
        """Calculates the free energy functional F[h]."""
        p = self.params
        dhdx = self.D @ h
        integrand = 0.5 * p['gamma'] * dhdx**2 + self.g1(h, p['a'], p['b'], p['c'], p['d'], p['k'])
        return np.sum(integrand) * self.dx

    def rhs(self, t, h):
        p = self.params
        h_xx = self.Laplacian @ h 
        mu =   - self.Pi1(h) - p['gamma'] * h_xx
        mu_x = self.D @ mu
        flux = self.D @ (p['Q'] * mu_x)
        source = p['g'] * h * (1 - h/ p['h_max'])

        return flux + source

    def solve(self, h0, T = 10, method = 'BDF', t_eval = None):
        if t_eval is None:
            t_eval = np.linspace(0, T, 5)
        sol = solve_ivp(self.rhs, [0, T], h0, t_eval = t_eval, method = method)
        return sol.t, sol.y


if __name__ == "__main__":
    start = time.time()
    model = OneD_Thin_Film_Model()

    h_init = model.setup_initial_conditions('gaussian')
    times, H = model.solve(h_init)

    figure_handler = fh.FigureHandler(model)
    figure_handler.plot_profiles(H, times)
    figure_handler.plot_binding_energy(model.g1)
    end = time.time()
    print(f"Run time: {end - start}")

    plt.show()