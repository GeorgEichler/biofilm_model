import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from helper_functions import find_first_k_minima
import figure_handler as fh
import time

class FFT_OneD_Thin_Film_Model:
    """
    Class to solve the 1D thin-film equation using fast Fourier transform
    with a implicit-explicit Euler scheme
    """

    def __init__(self, **kwargs):
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

        # Default parameters
        self.params = {
            'L': 50, 'N': 1000, 'gamma': 0.5, 'h_max': 5, 'g': 0.1,
            'a': 0.1, 'b': np.pi/2, 'c': 1, 'd': 0.02, 'k': 2*np.pi
        }
        self.params.update(kwargs)
        self._setup_grid_and_fft()


        # Calulate the first minima of the binding potential
        min, _ = find_first_k_minima(1, self.g1)
        self.h0 = min[0]

    def _setup_grid_and_fft(self):
        p = self.params
        self.dx = p['L'] / p['N']
        self.x = (np.arange(1, p['N'] + 1) - 0.5) * self.dx

        # wavenumbers for FFT
        k = 2*np.pi * fftfreq(p['N'], d = self.dx)
        self.k2 = k**2
        self.k4 = k**4

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
    
    def growth_term(self, h):
        p = self.params
        return p['g'] * (h - self.h0) * (1 - (h - self.h0)/p['h_max'])
    
    def time_step(self, h, dt):
        p = self.params

        # calculate explicit part in real space
        pi_h = self.Pi1(h)
        growth_h = self.growth_term(h)

        # Fourier transform
        pi_hat = fft(pi_h)
        growth_hat = fft(growth_h)

        # get explicit part -laplacian(Pi) + growth -> k2*pi_hat + growth_hat
        N_hat = self.k2 * pi_hat + growth_hat

        # solve biharmonic term implicitely
        h_hat = fft(h)
        denom = 1 + dt * p['gamma'] * self.k4
        h_hat_new = (h_hat + dt * N_hat) / denom

        # transform back to real space
        return ifft(h_hat_new).real

    def solve(self, h0, T, t_eval, dt = 0.1):
        h = h0.copy()
        num_steps = int(T / dt)
        t_eval = np.asarray(t_eval)

        t_snapshots = [0.0]
        h_snapshots = [h.copy()]

        # Calculate indices from simulation
        raw_indices = []

        for t_e in t_eval:
            # Time at step i is t = (i + 1) * dt (0 indication)
            # for i + 1 be closest integer to t_e /dt take i = round(t_e/dt) - 1
            step_idx = int(np.round(t_e / dt)) - 1

            if 0 <= step_idx < num_steps:
                raw_indices.append(step_idx)
        
        # create sorted list of unique indices
        target_indices = sorted(list(set(raw_indices)))
        target_ptr = 0

        start = time.time()
        print(f"Start integration using spectral methods in [0, {T}]...")
        for i in range(num_steps):
            # perform one time step
            h = self.time_step(h, dt)

            # check if we want a snapshot
            if target_ptr < len(target_indices) and i == target_indices[target_ptr]:
                current_t = (i + 1) * dt
                t_snapshots.append(current_t)
                h_snapshots.append(h.copy())
                target_ptr +=1
        end = time.time()
        print(f"Integration finished in {end - start}s.")

        results = {
            'times': np.array(t_snapshots),
            'H': np.array(h_snapshots)
        }

        return results
        
if __name__ == "__main__":
    # Simulation parameters
    T = 10.0
    dt = 0.1
    t_eval = np.linspace(0, T, 5)

    model = FFT_OneD_Thin_Film_Model()
    h0 = model.setup_initial_conditions('gaussian')


    results = model.solve(h0, T, t_eval)
    times = results['times']
    H = results['H']
    h_mins, g_mins = find_first_k_minima(
        k_minima = 5,
        f = model.g1
    )

    figure_handler = fh.FigureHandler(model)
    figure_handler.plot_profiles(H, times, pot_minima = h_mins)
    
    plt.show()