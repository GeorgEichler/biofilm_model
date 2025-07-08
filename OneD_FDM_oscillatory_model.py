import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
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
            'L': 50, 'N': 1000, 'Q': 1, 'gamma': 0.5, 'h_max': 5, 'g': 0.1,
            'a': 0.1, 'b': np.pi/2, 'c': 1.0, 'd': 0.02, 'k': 2*np.pi
        }

        # Update parameters with possible user-provided arguments
        self.params.update(kwargs)

        self._setup_grid_and_operators()

        # Calulate the first minima of the binding potential
        min, _ = find_first_k_minima(1, self.g1)
        self.h0 = min[0]

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

    # Pre-defined initial conditions
    def setup_initial_conditions(self, init_type):
        L = self.params['L']

        if init_type == 'gaussian':
            #h_init = 0.5 + 5 * np.exp(-(self.x - L/2)**2/0.1)
            h_init = 0.22 + 0.1 * np.exp(-(self.x - L/2)**2/10)
        elif init_type == 'constant':
            h_init = np.ones_like(self.x)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        
        return h_init

    # Define binding energies and corresponding disjoint pressures
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
        """Calculates the free energy functional F[h]
        Args:
            h (np.ndarray): array of height thickness
        Returns:
            Contribution of surface and potential energy
        """
        p = self.params
        dhdx = self.D @ h
        surface_energy = 0.5 * p['gamma'] * dhdx**2
        potential = self.g1(h)
        return [np.sum(surface_energy) * self.dx, np.sum(potential) * self.dx]

    # Right hand side of PDE
    def rhs(self, t, h):
        p = self.params
        h_xx = self.Laplacian @ h 
        mu =   - self.Pi1(h) - p['gamma'] * h_xx
        mu_x = self.D @ mu
        flux = self.D @ (p['Q'] * mu_x)
        source = p['g'] * (h - self.h0) * (1 - (h - self.h0) / p['h_max'])

        return flux + source

    def solve(self, h0, T = 10, method = 'BDF', t_eval = None):
        if t_eval is None:
            t_eval = np.linspace(0, T, 5)
        sol = solve_ivp(self.rhs, [0, T], h0, t_eval = t_eval, method = method)
        return sol.t, sol.y

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


if __name__ == "__main__":
    start = time.time()
    params = {'a': 1, 'gamma': 0.5}
    T = 10
    model = OneD_Thin_Film_Model(**params)
    t_eval = np.linspace(0, T, 5)
    t_plot = np.linspace(0, T, 5)

    h_init = model.setup_initial_conditions('gaussian')
    times, H = model.solve(h_init, T = T, t_eval = t_eval)

    figure_handler = fh.FigureHandler(model)
    h_mins, g1_mins = find_first_k_minima(
        k_minima=5, 
        f = model.g1
    )
    figure_handler.plot_profiles(H, t_plot, pot_minima=h_mins)
    figure_handler.plot_binding_energy(model.g1)

    print(f"Minima of $g_1$ are found at {h_mins} \n with values {g1_mins}.")
    figure_handler.plot_free_energy(H, times)
    end = time.time()
    print(f"Run time: {end - start}")

    plt.show()