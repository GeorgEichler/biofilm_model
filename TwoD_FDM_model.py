import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.integrate import solve_ivp
import time

from helper_functions import find_first_k_minima
from solver_wrapper import SolveIVPProgressWrapper
from numba import njit

@njit
def _rhs_stencil_numba(h, Nx, Ny, dx, dy, epsilon, g, h_max, ha, a, b, c, d, e, k, hf):
    """
    Compute RHS = Δμ + growth on a 2D periodic grid using 5-point Laplacian.
    h: flattened (Ny*Nx,) array, row-major order (y,x)
    """
    H = h.reshape(Ny, Nx)

    # Working arrays for intermediate calculations
    mu = np.empty_like(H)
    lap_h = np.empty_like(H)

    dx2 = dx * dx
    dy2 = dy * dy

    # --- First Pass: Calculate Laplacian of h and the chemical potential mu ---
    # mu = -ε Π(h) - Δh
    for j in range(Ny):
        jm = (j - 1 + Ny) % Ny
        jp = (j + 1) % Ny
        for i in range(Nx):
            im = (i - 1 + Nx) % Nx
            ip = (i + 1) % Nx

            h_ij = H[j, i]
            
            # 5-point Laplacian of h
            lap = (H[j, ip] - 2.0 * h_ij + H[j, im]) / dx2 + (H[jp, i] - 2.0 * h_ij + H[jm, i]) / dy2
            lap_h[j, i] = lap

            # Disjoining pressure Π(h)
            Pi = a * np.exp(-h_ij / c) * (k * np.sin(h_ij * k + b) + (1.0 / c) * np.cos(h_ij * k + b)) + (d / e) * np.exp(-h_ij / e)
            
            mu[j, i] = -epsilon * Pi - lap

    # --- Second Pass: Calculate Laplacian of mu ---
    # This gives the biharmonic/flux term
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

    # --- Third Pass: Calculate the local growth term ---
    growth = np.empty_like(H)
    for j in range(Ny):
        for i in range(Nx):
            hij = H[j, i]
            # NOTE: The original code had a hardcoded value 0.1. 
            # It's better to pass this as a parameter, let's call it hf.
            growth[j, i] = g * (hij - ha) * (1.0 - hij / h_max) * (1.0 - np.exp(hf - hij))

    return (flux + growth).ravel()

class FDM_TwoD_ThinFilm_Model:
    """
    Two dimensional thin film model with periodic BC on a uniform grid
    """
    def __init__(self, **kwargs):
        self.params = {
            'L': 25, 'N': 128, 'epsilon': 1, 'g': 1, 'h_max': 5, 'ha': 0.8, 'hf': 0.1,
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
    
    def _rhs_stencil(self, t, h_vec):
        p = self.params
        return _rhs_stencil_numba(
            h_vec, self.N, self.N, self.dx, self.dx,
            p['epsilon'], p['g'], p['h_max'], self.ha,
            p['a'], p['b'], p['c'], p['d'], p['e'], p['k'], p['hf']
        )
    
    def _laplacian_roll(self, H, dx2, dy2, out=None):
        # 5-point periodic Laplacian using vectorized rolls (no allocations if out provided)
        if out is None:
            out = (np.roll(H, -1, axis=1) + np.roll(H, 1, axis=1) - 2.0*H)/dx2
            out += (np.roll(H, -1, axis=0) + np.roll(H, 1, axis=0) - 2.0*H)/dy2
            return out
        else:
            np.add((np.roll(H, -1, axis=1) + np.roll(H, 1, axis=1) - 2.0*H)/dx2,
                (np.roll(H, -1, axis=0) + np.roll(H, 1, axis=0) - 2.0*H)/dy2,
                out=out)
            return out

    def _rhs_roll(self, t, h_vec):
        p = self.params
        N  = p['N']; dx = self.dx; dy = self.dx
        dx2 = dx*dx; dy2 = dy*dy

        H = h_vec.reshape(N, N)

        # reuse preallocated work arrays to avoid heap churn
        if not hasattr(self, "_buf"):
            self._buf = {
                "lap_h": np.empty_like(H),
                "mu"   : np.empty_like(H),
                "flux" : np.empty_like(H),
            }
        buf = self._buf

        # lap_h = Δh
        self._laplacian_roll(H, dx2, dy2, out=buf["lap_h"])

        # mu = -ε Π(h) - Δh
        Pi_h = self.Pi(H)                        # elementwise, vectorized
        np.multiply(-p['epsilon'], Pi_h, out=buf["mu"])
        buf["mu"] -= buf["lap_h"]

        # flux = Δμ
        self._laplacian_roll(buf["mu"], dx2, dy2, out=buf["flux"])

        # growth
        growth = self.growth_term(H)             # elementwise

        buf["flux"] += growth                    # in-place
        return buf["flux"].ravel()

    def _jac_sparsity_13pt(self):
        # builds a block-circulant sparsity for a 13–25 point stencil with periodic wrap
        N = self.N
        M = N*N
        from scipy.sparse import diags

        # offsets expressed in flattened indexing (row-major: index = y*N + x)
        offs = set([0, 1, -1, N, -N, 2, -2, 2*N, -2*N, N+1, N-1, -N+1, -N-1])

        # turn offsets into diagonals (wrap-around handled by adding a few more offsets)
        # Add also ±(2N±1) to be conservative if you want a 25-pt pattern:
        # offs.update([2*N+1, 2*N-1, -2*N+1, -2*N-1])

        data = []
        diags_idx = []
        for o in offs:
            diags_idx.append(o)
            data.append(np.ones(M))
        S = diags(data, diags_idx, shape=(M, M), format='csr')
        return S

    def solve(self, h0_vec, T = 100, method = 'LSODA', t_eval = None):
        rhs = SolveIVPProgressWrapper(self._rhs_roll, T)
        js = self._jac_sparsity_13pt()

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
    t, H = model.solve(h_init, T = T, t_eval = t_eval)

    N = model.N
    fig, axes = plt.subplots(1, len(t_eval), figsize=(3.4 * len(t_eval), 3.4), constrained_layout=True)
    for ax, ti, hi in zip(axes, t, H.T):
        im = ax.imshow(hi.reshape(N, N), origin='lower', extent=[0, model.L, 0, model.L])
        ax.set_title(f"t={ti:.0f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, shrink=0.8)


    plt.show()
