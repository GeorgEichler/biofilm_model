import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.integrate import solve_ivp
import time

from helper_functions import find_first_k_minima
from solver_wrapper import SolveIVPProgressWrapper
from numba import njit

@njit
def _rhs_stencil_numba(h, Nx, Ny, dx, dy, epsilon, g, h_max, h0, ha, a, b, c, d, e, k):
    """Compute RHS = Δμ + growth on a 2D periodic grid using 5-point Laplacian.
    h: flattened (Ny*Nx,) array, row-major order (y,x)
    """
    H = h.reshape(Ny, Nx)


    # working arrays
    mu = np.empty_like(H)
    lap_h = np.empty_like(H)


    dx2 = dx * dx
    dy2 = dy * dy


    # First Laplacian of h and mu = -ε Π(h) - Δh
    for j in range(Ny):
        jm = (j - 1 + Ny) % Ny
        jp = (j + 1) % Ny
        for i in range(Nx):
            im = (i - 1 + Nx) % Nx
            ip = (i + 1) % Nx


            h_ij = H[j, i]
            # 5-point Laplacian
            lap = (H[j, ip] - 2.0 * h_ij + H[j, im]) / dx2 + (H[jp, i] - 2.0 * h_ij + H[jm, i]) / dy2
            lap_h[j, i] = lap


            # disjoining pressure Π(h)
            Pi = a * np.exp(-h_ij / c) * (k * np.sin(h_ij * k + b) + (1.0 / c) * np.cos(h_ij * k + b)) + (d / e) * np.exp(-h_ij / e)
            mu[j, i] = -epsilon * Pi - lap


    # Second Laplacian: Δμ
    flux = np.empty_like(H)
    for j in range(Ny):
        jm = (j - 1 + Ny) % Ny
        jp = (j + 1) % Ny
        for i in range(Nx):
            im = (i - 1 + Nx) % Nx
            ip = (i + 1) % Nx
            mu_ij = mu[j, i]
            lap_mu = (mu[j, ip] - 2.0 * mu_ij + mu[j, im]) / dx2 + (mu[jp, i] - 2.0 * mu_ij + mu[jm, i]) / dy2
            flux[j, i] = lap_mu


    # Local growth term
    growth = np.empty_like(H)
    for j in range(Ny):
        for i in range(Nx):
            hij = H[j, i]
            growth[j, i] = g * (hij - ha) * (1.0 - hij / h_max) * (1.0 - np.exp(0.1 - hij))


    return (flux + growth).ravel()

class FDM_TwoD_ThinFilm_Model:
    """
    Two dimensional thin film model with periodic BC on a uniform grid
    """
    def __init__(self, **kwargs):
        self.params = {
            'L': 10, 'N': 64, 'epsilon': 1, 'g': 1, 'h_max': 1, 'ha': 0.8,
            'a': 0.5, 'b': np.pi, 'c': 1.0, 'd': 10.0, 'e': 0.01, 'k': 2*np.pi,
            'amplitude': 1.0, 'var': 10 
        }
        self.params.update(kwargs)
        p = self.params

        self.N = p['N']
        self.L = p['L']
        self.dx = p['L'] / p['N']
        x = (np.arange(1, p['N'] + 1) - 0.5) * self.dx
        y = (np.arange(1, p['N'] + 1) - 0.5) * self.dx
        self.x, self.y = np.meshgrid(x, y)

        self._setup_fd_operators()

        # Calculate equilibrium heights h0 and h1
        minima, _ = find_first_k_minima(2, self.f)
        self.h0 = minima[0]
        self.h1 = minima[1]
        self.ha = p['ha'] # activation point

    def _setup_fd_operators(self):
        N = self.params['N']

        # 1D periodic first derivative
        D = diags([-1, 1], [-1, 1], shape=(N, N), format = 'lil')
        D[0, -1] = -1
        D[-1, 0] = 1
        D = (D / (2.0 * self.dx)).tocsr()

        # 1D periodic Laplacian
        L = diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format='lil')
        L[0, -1] = 1
        L[-1, 0] = 1
        L = (L / (self.dx ** 2)).tocsr()

        I = eye(N, format = 'csr')
        self.D = kron(I, D, format='csr')
        self.Laplacian = kron(I, L, format='csr') + kron(L, I, format='csr')


    def setup_initial_conditions(self, init_type='gaussian'):
        p = self.params
        N = p['N']
        H = np.empty((N, N), dtype=float)
        if init_type == 'gaussian':
            c = 0.5 * p['L']
            r2 = (self.x - c) ** 2 + (self.y - c) ** 2
            H[:, :] = (self.h0 + 0.01) + p['amplitude'] * np.exp(-r2 / p['var'])
        elif init_type == 'constant':
            H[:, :] = self.h0 + 0.5
        elif init_type == 'cap':
            c = 0.5 * p['L']
            H[:, :] = np.maximum(self.h0 + 0.01, p['amplitude'] - (1.0 / p['var']) * ((self.x - c) ** 2 + (self.y - c) ** 2))
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")
        return H.ravel()

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

        growth = p['g'] * (h - self.ha) * (1 - h/p['h_max']) * (1 - np.exp( 0.1 - h ))

        #growth = np.maximum(growth, 0) # alternative growth term with truncated growth

        return growth
    
    def _rhs_sparse(self, t, h_vec):
        p = self.params
        N = p['N']
        H = h_vec.reshape(N, N)

        Pi_h = self.Pi(H)
        lap_h = (self.Laplacian @ h_vec).reshape(N, N)
        mu = - p['epsilon'] * Pi_h - lap_h

        flux = (self.Laplacian @mu.ravel()).reshape(N, N)

        growth = self.growth_term(H)

        return (flux + growth).ravel()
    
    def solve(self, h0_vec, T = 100, method = 'LSODA', t_eval = None):
        rhs = SolveIVPProgressWrapper(self._rhs_sparse, T)

        print(f"Start 2D integration ({self.N}×{self.N}) using {method} on [0, {T}]...")
        start = time.time()
        sol = solve_ivp(
            rhs,
            [0, T],
            h0_vec,
            method = method,
            t_eval = t_eval
        )
        end = time.time()
        print(f"Integration finished in {end - start:.3f}s. Status={sol.status} (1=event, 0=end)")
        return sol.t, sol.y

if __name__ == '__main__':

    model = FDM_TwoD_ThinFilm_Model()
    h_init = model.setup_initial_conditions('gaussian')
    T = 10

    t_eval = np.linspace(0, T, 5)
    t, H = model.solve(h_init, T = T, t_eval = t_eval, method = 'BDF')

    N = model.N
    fig, axes = plt.subplots(1, len(t_eval), figsize=(3.4 * len(t_eval), 3.4), constrained_layout=True)
    for ax, ti, hi in zip(axes, t, H.T):
        im = ax.imshow(hi.reshape(N, N), origin='lower', extent=[0, model.L, 0, model.L])
        ax.set_title(f"t={ti:.0f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, shrink=0.8)


    plt.show()
