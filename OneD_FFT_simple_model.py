import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from helper_functions import find_first_k_minima
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
            'L': 10, 'N': 128, 'gamma': 0.5, 'h_max': 5, 'g': 0.1,
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
        
if __name__ == "__main__":
    # Simulation parameters
    T_final = 10.0
    dt = 0.1

    model = FFT_OneD_Thin_Film_Model(N=512, L=10.0)
    h = model.setup_initial_conditions('gaussian')

    num_steps = int(T_final/dt)
    print(f"Running {num_steps} steps (dt={dt})...")

    # Set up live plotting
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(model.x, h)
    ax.set_ylim(h.min()-0.1, h.max()+0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.set_title("t = 0.00")

    start = time.time()
    plot_every = 20

    for i in range(num_steps):
        h = model.time_step(h, dt)
        if (i+1) % plot_every == 0:
            t = (i+1)*dt
            line.set_ydata(h)
            ax.set_title(f"t = {t:.2f}")
            ax.set_ylim(h.min(), h.max())
            plt.pause(0.01)

    plt.ioff()
    end = time.time()

    print(f"Done in {end-start:.2f}s")

    # Final snapshot
    plt.figure()
    plt.plot(model.x, h, '-')
    plt.xlabel("x")
    plt.ylabel("h(x)")
    plt.title(f"Final State at t={T_final}")
    plt.show()