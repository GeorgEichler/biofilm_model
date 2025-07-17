import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import sparse
from helper_functions import find_first_k_minima
import time
import numba

@numba.jit(nopython = True, cache = True)
def _numba_laplacian(Z, delta_sq):
    """
    Numba-Jit compiled 5-point stencil Laplacian with periodic boundary conditions
    """
    Nx, Ny = Z.shape
    lap_Z = np.empty_like(Z)
    for i in range(Nx):
        for j in range(Ny):
            Z_n = Z[(i + 1) % Nx, j]
            Z_s = Z[(i - 1 + Nx) % Nx, j]
            Z_e = Z[i, (j + 1) % Ny]
            Z_w = Z[i, (j - 1 + Ny) % Ny]
            lap_Z[i, j] = (Z_n + Z_s + Z_e + Z_w - 4 * Z[i, j]) / delta_sq
    return lap_Z

@numba.jit(nopython=True, cache=True)
def numba_rhs(h, shape, delta_sq, gamma, g, h0, h_max, a, b, c, d, k):
    """
    The full RHS calculation, compiled by Numba
    """
    h = h.reshape(shape)
    
    # Binding potential derivative (Pi1)
    pi_h = a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))
    
    # Chemical potential (mu)
    lap_h = _numba_laplacian(h, delta_sq)
    mu = -pi_h - gamma * lap_h
    
    # Flux term
    flux = _numba_laplacian(mu, delta_sq)
    
    # Source term
    source = g * (h - h0) * (1 - (h - h0) / h_max)
    
    dhdt = flux + source
    return dhdt.flatten()

class TwoD_Thin_Film_Model:
    """
    A class to set up the two dimensional thin film model
    """

    def __init__(self, **kwargs):
        """
        Kwargs:
            Lx, Ly (float): Domain length [0, Lx] x [0, Ly]
            Nx, Ny (int): Number of grid points
            gamma (float): Surface tension
            h_max (float): maximal film height for the growth term
            g (float): coefficient of logistic growth, qoutient of growth and diffusion coefficient
            a, b, c, d, k (float): Parameter for the binding potential
            h_init_type (str): Type of inital condition
        """
        
        # Default values
        self.params = {
            'Lx': 10, 'Ly': 10, 'Nx': 100, 'Ny': 100, 'gamma': 0.5, 'h_max': 5, 'g': 0.1,
            'a': 0.1, 'b': np.pi/2, 'c': 1.0, 'd': 0.02, 'k': 2*np.pi,
            'amplitude': 2, 'var': 10
        }

        # Update parameters with possible user-provided arguments
        self.params.update(kwargs)

        self._setup_grid()

        # Calulate the first minima of the binding potential
        min, _ = find_first_k_minima(1, self.g1)
        self.h0 = min[0]

    #Assumes an uniform grid
    def _setup_grid(self):
        p = self.params
        self.dx = p['Lx'] / p['Nx']
        self.dy = p['Ly'] / p['Ny']
        
        # For simplicity we use a square grid
        if self.dx != self.dy:
            print(f"Warning: dx ({self.dx}) != dy ({self.dy}). Using average value for operators.")
        self.delta = self.dx
        self.delta_sq = self.delta**2 

        # Store grid shape for easy reshaping
        self.shape = (p['Nx'], p['Ny'])

        x = (np.arange(1, p['Nx']+ 1) - 0.5) * self.dx
        y = (np.arange(1, p['Ny']+ 1) - 0.5) * self.dy

        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

    def setup_initial_conditions(self, init_type='gaussian'):
        """
        Generates a 2D initial condition. This function is UNCHANGED, but it will
        now use the new cell-centered self.X and self.Y coordinates.
        """
        p = self.params
        Lx = p['Lx']
        Ly = p['Ly']
        amplitude = p['amplitude']
        var = p['var']

        if init_type == 'gaussian':
            # A single Gaussian bump in the center
            h_init = self.h0 + amplitude * np.exp(-((self.X - Lx/2)**2 + (self.Y - Ly/2)**2) / var)
        elif init_type == 'random':
            # Small random perturbations around h0
            h_init = self.h0 + 0.05 * (np.random.rand(*self.shape) - 0.5)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        
        return h_init
    
    def g1(self, h):
        p = self.params
        a, b, c, d, k = p['a'], p['b'], p['c'], p['d'], p['k']
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-h/(2*c))

    def Pi1(self, h):
        p = self.params
        a, b, c, d, k = p['a'], p['b'], p['c'], p['d'], p['k']
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))
    

    def rhs(self, t, h_flat):
        """
        Calculate the right-hand side of the PDE
        """
        p = self.params

        return numba_rhs(h_flat, self.shape, self.delta_sq, p['gamma'], p['g'], self.h0, p['h_max'],
                         p['a'], p['b'], p['c'], p['d'], p['k']) 

    def solve(self, h0_2d, T=10, method='BDF', t_eval=None):
        """
        Solve the PDE using an ODE integrator
        """
        start_time = time.time()
        
        
        # --- Jacobian Sparsity Pattern Construction ---
        print("Constructing Jacobian sparsity pattern...")
        p = self.params
        N = p['Nx'] * p['Ny']
        
        # These are the offsets for a 9-point stencil (self + 8 neighbors)
        offsets = [0, 
                -1, 1, 
                -p['Nx'], p['Nx'],
                -(p['Nx']+1), p['Nx']+1,
                -(p['Nx']-1), p['Nx']-1]
        
        # THE FIX: Create a 2D array where each row is a full-length diagonal.
        # The shape is (number_of_diagonals, length_of_each_diagonal).
        num_diags = len(offsets)
        diagonals_data = np.ones((num_diags, N))

        # The spdiags function now receives a correctly shaped data array.
        jac_sparsity = sparse.spdiags(diagonals_data, offsets, N, N, format='csc')
        
        setup_end_time = time.time()
        print(f"Setup finished in {setup_end_time - start_time:.3f}s.")
        
        if t_eval is None:
            t_eval = np.linspace(0, T, 5)
            
        h0_flat = h0_2d.flatten()
        print("Compiling Numba functions (first call)...")
        _ = self.rhs(0, h0_flat)
        compile_end_time = time.time()
        print(f"Compilation finished in {compile_end_time - setup_end_time}s.")


        print(f"Start integration using {method} method in [0, {T}]...")
        sol = solve_ivp(self.rhs, [0, T], h0_flat, t_eval=t_eval, method=method,
                        jac_sparsity = jac_sparsity)
        
        end_time = time.time()
        print(f"Integration finished in {end_time - compile_end_time:.3f}s.")
        
        H_2d = sol.y.T.reshape(len(t_eval), *self.shape)
        return sol.t, H_2d

if __name__ == "__main__":
    # --- Simulation Setup ---
    T_final = 1
    
    model = TwoD_Thin_Film_Model()
    
    t_points = np.linspace(0, T_final, 5)
    h_initial = model.setup_initial_conditions(init_type='gaussian')
    
    # Solve the equation
    times, H_solution = model.solve(h_initial, T=T_final, t_eval=t_points)
    
    # --- Visualization ---
    print("Plotting results...")
    fig, axes = plt.subplots(1, len(times), figsize=(15, 4))
    
    vmin = H_solution.min()
    vmax = H_solution.max()
    
    for i, ax in enumerate(axes.flat):
        # Use pcolormesh for cell-centered data for more accurate axis labels,
        # or imshow with extent for better performance.
        im = ax.imshow(H_solution[i].T, cmap='viridis', vmin=vmin, vmax=vmax,
                       extent=[0, model.params['Lx'], 0, model.params['Ly']],
                       origin='lower', interpolation='nearest')
        ax.set_title(f"t = {times[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Biofilm Height h(x,y)")
    fig.suptitle("2D Biofilm Evolution (Cell-Centered Grid)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()