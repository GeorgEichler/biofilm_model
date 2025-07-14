import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import sparse
from helper_functions import find_first_k_minima
import time
import numba


# --- Numba-Optimized Helper Functions ---

@numba.jit(nopython=True, cache=True)
def _numba_laplacian(Z, delta_sq):
    """
    Numba-JIT compiled 5-point stencil Laplacian with periodic boundary conditions.
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
    The full RHS calculation, compiled by Numba.
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
    A class to set up the two dimensional thin film model.
    """
    def __init__(self, **kwargs):
        self.params = {
            'Lx': 10, 'Ly': 10, 'Nx': 128, 'Ny': 128, 'gamma': 0.5, 'h_max': 5, 'g': 0.1,
            'a': 0.1, 'b': np.pi/2, 'c': 1.0, 'd': 0.02, 'k': 2*np.pi
        }
        self.params.update(kwargs)
        self._setup_grid()
        min_val, _ = find_first_k_minima(1, self.g1)
        self.h0 = min_val[0]

    def _setup_grid(self):
        p = self.params
        self.dx = p['Lx'] / p['Nx']
        self.dy = p['Ly'] / p['Ny']
        if self.dx != self.dy:
            print(f"Warning: dx ({self.dx}) != dy ({self.dy}). Using dx for spacing.")
        self.delta = self.dx 
        self.delta_sq = self.delta**2
        self.shape = (p['Nx'], p['Ny'])
        x = (np.arange(1, p['Nx']+ 1) - 0.5) * self.dx
        y = (np.arange(1, p['Ny']+ 1) - 0.5) * self.dy
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')

    def setup_initial_conditions(self, init_type='gaussian'):
        p = self.params
        if init_type == 'gaussian':
            h_init = self.h0 + 2 * np.exp(-((self.X - p['Lx']/2)**2 + (self.Y - p['Ly']/2)**2) / 10)
        elif init_type == 'random':
            h_init = self.h0 + 0.05 * (np.random.rand(*self.shape) - 0.5)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        return h_init
    
    def g1(self, h):
        p = self.params
        a, b, c, d, k = p['a'], p['b'], p['c'], p['d'], p['k']
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-h/(2*c))

    def rhs(self, t, h_flat):
        p = self.params
        return numba_rhs(h_flat, self.shape, self.delta_sq, p['gamma'], p['g'], self.h0, p['h_max'],
                         p['a'], p['b'], p['c'], p['d'], p['k'])

    # --- ANALYTICAL JACOBIAN METHODS ---
    def Pi1_prime(self, h):
        """Calculates the derivative of Pi1(h) with respect to h."""
        p = self.params
        a, b, c, d, k = p['a'], p['b'], p['c'], p['d'], p['k']
        
        term1 = (k**2 * np.cos(h*k + b) - (2*k/c) * np.sin(h*k + b) - (1/c**2) * np.cos(h*k + b))
        term2 = a * np.exp(-h/c) * term1
        term3 = -(d / (4 * c**2)) * np.exp(-h / (2*c))
        
        return term2 + term3

    def S_prime(self, h):
        """Calculates the derivative of the source term S(h) with respect to h."""
        p = self.params
        return p['g'] * (1 - 2 * (h - self.h0) / p['h_max'])

    def get_jacobian_func(self):
        """
        Pre-computes constant parts of the Jacobian and returns a function
        that calculates the Jacobian for a given state h.
        """
        p = self.params
        Nx, Ny = p['Nx'], p['Ny']
        N = Nx * Ny
        delta2 = self.delta**2

        # Create the sparse Laplacian matrix L with periodic boundaries
        main_diag = np.full(N, -4.0)
        side_diag = np.ones(N)
        up_down_diag = np.ones(N)
        
        side_diag[Nx-1::Nx] = 0 # Zero out connections at right-to-left wrap-around point
        
        diagonals = [main_diag, side_diag[:-1], side_diag[:-1], up_down_diag[:-Nx], up_down_diag[:-Nx]]
        offsets = [0, 1, -1, Nx, -Nx]
        
        L = sparse.diags(diagonals, offsets, shape=(N, N), format='csc')
        
        # Add periodic connections explicitly
        # Left-right periodic connections
        lr_indices = np.arange(0, N, Nx)
        L[lr_indices, lr_indices + (Nx - 1)] = 1
        L[lr_indices + (Nx - 1), lr_indices] = 1
        
        # Top-bottom periodic connections
        tb_indices = np.arange(Nx)
        L[tb_indices, tb_indices + N - Nx] = 1
        L[tb_indices + N - Nx, tb_indices] = 1
        
        L /= delta2
        
        # Pre-compute L-squared
        L_squared = L @ L
        
        # This is the function that will be called by the solver
        def jac_func(t, h_flat):
            h = h_flat.reshape(self.shape)
            
            pi1_prime_vals = self.Pi1_prime(h).flatten()
            s_prime_vals = self.S_prime(h).flatten()
            
            Pi1_prime_diag = sparse.diags(pi1_prime_vals, format='csc')
            S_prime_diag = sparse.diags(s_prime_vals, format='csc')
            
            J = -L @ Pi1_prime_diag - p['gamma'] * L_squared + S_prime_diag
            return J

        print("Analytical Jacobian function has been created.")
        return jac_func

    def solve(self, h0_2d, T=10, method='BDF', t_eval=None):
        """
        Solve the PDE using an ODE integrator with a provided analytical Jacobian.
        """
        start_time = time.time()
        
        if t_eval is None:
            t_eval = np.linspace(0, T, 10)
            
        h0_flat = h0_2d.flatten()
        
        # Generate the Jacobian function
        jacobian_func = self.get_jacobian_func()
        
        setup_end_time = time.time()
        print(f"Setup finished in {setup_end_time - start_time:.3f}s.")

        # Warm-up call for Numba
        print("Compiling Numba functions (first call)...")
        _ = self.rhs(0, h0_flat)
        compile_end_time = time.time()
        print(f"Compilation finished in {compile_end_time - setup_end_time:.3f}s.")

        print(f"Start integration using {method} method in [0, {T}]...")
        # Pass the analytical Jacobian function to the solver
        sol = solve_ivp(self.rhs, [0, T], h0_flat, t_eval=t_eval, method=method, 
                        jac=jacobian_func)
        
        end_time = time.time()
        print(f"Integration finished in {end_time - compile_end_time:.3f}s.")
        
        H_2d = sol.y.T.reshape(len(t_eval), *self.shape)
        return sol.t, H_2d

if __name__ == "__main__":
    T_final = 10
    
    # Using a 128x128 grid to really see the performance difference
    model = TwoD_Thin_Film_Model()
    
    t_points = np.linspace(0, T_final, 5)
    h_initial = model.setup_initial_conditions(init_type='gaussian')
    
    times, H_solution = model.solve(h_initial, T=T_final, t_eval=t_points)
    
    print("Plotting results...")
    fig, axes = plt.subplots(1, len(times), figsize=(15, 4), constrained_layout=True)
    vmin = H_solution.min()
    vmax = H_solution.max()
    
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(H_solution[i].T, cmap='viridis', vmin=vmin, vmax=vmax,
                       extent=[0, model.params['Lx'], 0, model.params['Ly']],
                       origin='lower', interpolation='nearest')
        ax.set_title(f"t = {times[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Biofilm Height h(x,y)")
    fig.suptitle("2D Biofilm Evolution (Fully Optimized with Analytical Jacobian)")
    plt.show()