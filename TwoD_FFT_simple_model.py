import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.fft import fft2, ifft2, fftfreq
import matplotlib.colors as mcolors
from helper_functions import find_first_k_minima

class FastTwoD_Thin_Film_Model:
    """
    A class to solve the 2D thin-film equation using a fast,
    semi-implicit Fourier-spectral method.
    """

    def __init__(self, **kwargs):
        # Default parameters
        self.params = {
            'Lx': 50, 'Ly': 50, 'Nx': 256, 'Ny': 256, 'gamma': 0.5, 
            'h_max': 5, 'g': 0.1, 'a': 0.1, 'b': np.pi/2, 'c': 1.0, 
            'd': 0.02, 'k': 2*np.pi, 'amplitude': 2, 'var': 10
        }
        self.params.update(kwargs)
        self._setup_grid_and_fft()

        # Calulate the first minima of the binding potential
        min, _ = find_first_k_minima(1, self.g1)
        self.h0 = min[0]

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
        Lx = p['Lx']
        Ly = p['Ly']
        amplitude = p['amplitude']
        var = p['var']

        if init_type == 'gaussian':
            h_init = self.h0 + amplitude * np.exp(-((self.X - Lx/2)**2 + (self.Y - Ly/2)**2) / var)
        elif init_type == 'random':
            h_init = self.h0 + 0.05 * (np.random.rand(*self.shape) - 0.5)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        return h_init

    # Binding potential and disjoint pressure
    def g1(self, h):
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; k = p['k']
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-h/(2*c))

    def Pi1(self, h):
        p = self.params
        a, b, c, d, k = p['a'], p['b'], p['c'], p['d'], p['k']
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))

    def growth_term(self, h):
        p = self.params
        growth = p['g'] * (h - self.h0) * (1 - (h - self.h0) / p['h_max'])
        return np.where(h > self.h0, growth, 0.0)

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
    T_final = 50 
    dt = 0.1
    params = {'g': 0.1, 'gamma': 10}
    model = FastTwoD_Thin_Film_Model(**params)
    h = model.setup_initial_conditions(init_type='gaussian')
    
    num_steps = int(T_final / dt)
    print(f"Running simulation for {num_steps} steps with dt={dt}...")

    # Define the height thresholds that separate the layers.
    p = model.params
    num_layers = 5
    h_min, g_min = find_first_k_minima(k_minima = num_layers, f = model.g1)
    layer_thresholds = np.array(h_min)
    num_layers = len(layer_thresholds) - 1
    print(f"Layer thresholds set at: {np.round(layer_thresholds, 2)}")

    # Create a discrete colormap. We'll pick N colors for N layers
    #    'tab10', 'Set1', 'Accent' are good choices for distinct colors
    cmap = plt.get_cmap('viridis', num_layers)

    # Create a BoundaryNorm to map height values to our discrete colors
    norm = mcolors.BoundaryNorm(layer_thresholds, cmap.N)
    
    # --- Live Visualization ---
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    # We pass the raw height data `h` and the norm handles the color mapping.
    im = ax.imshow(h.T, cmap=cmap, norm=norm, origin='lower', 
                   extent=[0, p['Lx'], 0, p['Ly']])
    
    # Calculate the midpoint of each color band for tick placement
    tick_locs = layer_thresholds[:-1] + np.diff(layer_thresholds)/2
    cbar = fig.colorbar(im, ticks=tick_locs, label="Biofilm Layer")
    # Set the labels for each tick
    tick_labels = [f'Layer {i}' for i in range(num_layers)]
    cbar.set_ticklabels(tick_labels)
    
    ax.set_title(f"t = 0.00")
    
    start_time = time.time()
    
    plot_every = 10
    for i in range(num_steps):
        h = model.step(h, dt)
        
        if (i + 1) % plot_every == 0:
            current_sim_time = (i + 1) * dt
            im.set_data(h.T)

            # The `norm` object handles the color mapping consistently.
            ax.set_title(f"t = {current_sim_time:.2f}")
            plt.pause(0.01)

    end_time = time.time()
    plt.ioff()
    
    print(f"\nSimulation finished.")
    print(f"Total time for {num_steps} steps: {end_time - start_time:.2f} seconds.")
    
    # Show final state using the same layered plot style
    plt.figure(figsize=(7,7))
    plt.imshow(h.T, cmap=cmap, norm=norm, origin='lower',
               extent=[0, p['Lx'], 0, p['Ly']])
    
    # Re-create the custom colorbar for the final plot
    cbar = plt.colorbar(ticks=tick_locs, label="Biofilm Layer")
    cbar.set_ticklabels(tick_labels)
    
    plt.title(f"Final State at t = {T_final:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()