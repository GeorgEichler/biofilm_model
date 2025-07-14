import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.fft import fft2, ifft2, fftfreq

class FastTwoD_Thin_Film_Model:
    """
    A class to solve the 2D thin-film equation using a fast,
    semi-implicit Fourier-spectral method.
    """

    def __init__(self, **kwargs):
        # Default parameters
        self.params = {
            'Lx': 10, 'Ly': 10, 'Nx': 128, 'Ny': 128, 'gamma': 0.5, 
            'h_max': 5, 'g': 0.1, 'a': 0.1, 'b': np.pi/2, 'c': 1.0, 
            'd': 0.02, 'k': 2*np.pi, 'h0': 0.23
        }
        self.params.update(kwargs)
        self._setup_grid_and_fft()

    def _setup_grid_and_fft(self):
        """Sets up the grid and pre-computes Fourier space operators."""
        p = self.params
        self.dx = p['Lx'] / p['Nx']
        self.dy = p['Ly'] / p['Ny']
        self.shape = (p['Nx'], p['Ny'])

        # Create cell-centered coordinate arrays
        x = (np.arange(p['Nx']) + 0.5) * self.dx
        y = (np.arange(p['Ny']) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

        # --- Fourier Space Setup ---
        # Get wavenumbers for FFT
        kx = 2 * np.pi * fftfreq(p['Nx'], d=self.dx)
        ky = 2 * np.pi * fftfreq(p['Ny'], d=self.dy)
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

        # Laplacian operator in Fourier space: -|k|^2
        self.K2 = Kx**2 + Ky**2
        # Biharmonic operator in Fourier space: |k|^4
        self.K4 = self.K2**2

    def setup_initial_conditions(self, init_type='random'):
        """Generates a 2D initial condition."""
        p = self.params
        if init_type == 'gaussian':
            h_init = p['h0'] + 2 * np.exp(-((self.X - p['Lx']/2)**2 + (self.Y - p['Ly']/2)**2) / 10)
        elif init_type == 'random':
            h_init = p['h0'] + 0.05 * (np.random.rand(*self.shape) - 0.5)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        return h_init

    # --- Physics Functions (element-wise, no changes needed) ---
    def Pi1(self, h):
        p = self.params
        a, b, c, d, k = p['a'], p['b'], p['c'], p['d'], p['k']
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))

    def growth_term(self, h):
        p = self.params
        return p['g'] * (h - p['h0']) * (1 - (h - p['h0']) / p['h_max'])

    def step(self, h, dt):
        """
        Performs a single time step using the semi-implicit Fourier method.
        """
        p = self.params
        
        # --- 1. Calculate nonlinear/explicit part in REAL space ---
        pi_h = self.Pi1(h)
        growth = self.growth_term(h)
        
        # Transform to Fourier space
        pi_h_hat = fft2(pi_h)
        growth_hat = fft2(growth)
        
        # The nonlinear term is -laplacian(Pi) + growth
        # In Fourier space, this is -(-K2)*Pi_hat + growth_hat
        N_hat = self.K2 * pi_h_hat + growth_hat
        
        # --- 2. Apply the semi-implicit update rule in FOURIER space ---
        h_hat = fft2(h)
        
        # Denominator for the implicit solve: (1 + dt * gamma * K4)
        denominator = 1 + dt * p['gamma'] * self.K4
        
        # Update rule: h_hat_next = (h_hat + dt * N_hat) / denominator
        h_hat_next = (h_hat + dt * N_hat) / denominator
        
        # --- 3. Transform back to REAL space ---
        h_next = ifft2(h_hat_next).real
        
        return h_next

if __name__ == "__main__":
    # --- Simulation Setup ---
    T_final = 10
    dt = 0.1  # Can be 1000x larger than for a fully explicit method!
    
    model = FastTwoD_Thin_Film_Model()
    h = model.setup_initial_conditions(init_type='gaussian')
    
    num_steps = int(T_final / dt)
    print(f"Running simulation for {num_steps} steps with dt={dt}...")

    # --- Live Visualization ---
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(h.T, cmap='viridis', origin='lower', 
                   extent=[0, model.params['Lx'], 0, model.params['Ly']])
    cbar = fig.colorbar(im, label="Biofilm Height h(x,y)")
    ax.set_title(f"t = 0.00")
    
    start_time = time.time()
    
    plot_every = 20 # Update plot every 20 steps
    for i in range(num_steps):
        h = model.step(h, dt)
        
        if (i + 1) % plot_every == 0:
            current_sim_time = (i + 1) * dt
            im.set_data(h.T)
            im.set_clim(h.min(), h.max())
            ax.set_title(f"t = {current_sim_time:.2f}")
            plt.pause(0.01)

    end_time = time.time()
    plt.ioff()
    
    print(f"\nSimulation finished.")
    print(f"Total time for {num_steps} steps: {end_time - start_time:.2f} seconds.")
    
    # Show final state
    plt.figure(figsize=(7,7))
    plt.imshow(h.T, cmap='viridis', origin='lower',
               extent=[0, model.params['Lx'], 0, model.params['Ly']])
    plt.colorbar(label="Biofilm Height h(x,y)")
    plt.title(f"Final State at t = {T_final:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()