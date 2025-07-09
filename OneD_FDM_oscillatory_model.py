import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import figure_handler as fh
from helper_functions import find_first_k_minima
import time

class OneD_Thin_Film_Model:
    """
    A class to set up a one dimensional thin-film equation model
    """

    def __init__(self, use_fft = False, **kwargs):
        """
        Kwargs:
            L (float): Domain length [0, L]
            N (int): Number of grid points
            gamma (float): Surface tension
            h_max (float): maximal film height for the growth term
            g (float): coefficient of logistic growth, qoutient of growth and diffusion coefficient
            a, b, c, d, k (float): Parameter for the binding potential
            h_init_type (str): Type of inital condition
        """
        
        self.use_fft = use_fft
        # Default values
        self.params = {
            'L': 50, 'N': 1000, 'gamma': 0.5, 'h_max': 5, 'g': 0.1,
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

        # First derivative with periodic boundary conditions, also needed for free energy
        D = diags(diagonals=[-1,1], offsets=[-1,1], shape=(N, N), format= 'lil')
        D[0, -1] = -1
        D[-1, 0] = 1

        self.D = (D / (2 * self.dx)).asformat('csr')

        if self.use_fft:
            # Calculate FFT wavenumbers
            self.fft_k = 2 *np.pi * fftfreq(N, d = self.dx)
        
        else:
            # Second derivative with periodic boundary conditions
            Laplacian = diags(diagonals=[1,-2,1], offsets=[-1,0,1], shape = (N, N), format = 'lil')
            Laplacian[0, -1] = 1
            Laplacian[-1, 0] = 1

            self.Laplacian = (Laplacian / (self.dx**2)).asformat('csr')
            

    # Pre-defined initial conditions
    def setup_initial_conditions(self, init_type):
        L = self.params['L']

        if init_type == 'gaussian':
            h_init = (self.h0 + 0.01) + 5 * np.exp(-(self.x - L/2)**2/0.1)
        elif init_type == 'constant':
            h_init = np.ones_like(self.x)
        elif init_type == 'bump':
            h_init = self.h0 + 0.1 * np.exp(-(self.x - L/2)**2/10)
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

    def _rhs_fdm(self, t, h):
        """RHS for finite difference method"""
        p = self.params
        h_xx = self.Laplacian @ h 
        mu = - self.Pi1(h) - p['gamma'] * h_xx
        mu_x = self.D @ mu
        flux = self.D @ mu_x
        source = p['g'] * (h - self.h0) * (1 - (h - self.h0)/p['h_max'])
        return flux + source

    # Right hand side when using FFT
    def _rhs_fft(self, t, h):
        p = self.params
        # Compute nonlinear terms in real space
        pi_h = self.Pi1(h)
        source_h = p['g'] * (h - self.h0) * (1 - (h - self.h0) / p['h_max'])

        # Transform to Fourier space
        h_hat = fft(h)
        pi_h_hat = fft(pi_h)
        source_h_hat = fft(source_h)

        # Assemble RHS in Fourier space using d^n/dx^n -> (i*k)^n
        k2 = self.fft_k**2
        k4 = self.fft_k**4

        rhs_hat = (-p['gamma'] * k4 * h_hat + k2 * pi_h_hat) + source_h_hat

        return ifft(rhs_hat).real 

    # Right hand side of PDE
    def rhs(self, t, h):
        if self.use_fft:
            return self._rhs_fft(t, h)
        else:
            return self._rhs_fdm(t, h)
        
    # Good possible methods due to the stiffness are LSODA, BDF or Radau
    def solve(self, h0, T = 10, method = 'LSODA', t_eval = None):
        start = time.time()
        print(f"Start integration using {method} method in [0, {T}]...")
        if t_eval is None:
            t_eval = np.linspace(0, T, 5)
        sol = solve_ivp(self.rhs, [0, T], h0, t_eval = t_eval, method = method)
        end = time.time()
        print(f"Integration finished in {end - start:.3f}s.")
        return sol.t, sol.y


if __name__ == "__main__":
    params = {'a': 1, 'gamma': 0.5}
    T = 10
    model = OneD_Thin_Film_Model(use_fft=False,**params)
    t_eval = np.linspace(0, T, 5)
    t_plot = np.linspace(0, T, 5)

    h_init = model.setup_initial_conditions('gaussian')
    times, H = model.solve(h_init, T = T, t_eval = t_eval)

    figure_handler = fh.FigureHandler(model)
    figure_handler.plot_profiles(H, t_plot)

    """
    h_mins, g1_mins = find_first_k_minima(
        k_minima=5, 
        f = model.g1
    )
    figure_handler.plot_binding_energy(model.g1)
    print(f"Minima of g\u2081 are found at {h_mins} \n with values {g1_mins}.")
    figure_handler.plot_free_energy(H, times)
    """

    plt.show()