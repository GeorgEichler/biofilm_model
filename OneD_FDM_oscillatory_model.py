import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from config import OneDConfig
import figure_handler as fh

class OneD_Thin_Film_Model:
    """
    A class to set up a one dimensional thin-film equation model
    """

    def __init__(self, config: OneDConfig):
        self.config = config
        self.params = config.params


        self._setup_grid_and_operators()
        self._setup_initial_conditions()

    def _setup_grid_and_operators(self):
        N = self.params['N']
        L = self.params['L']

        self.dx = L / N
        self.x = (np.arange(1, N + 1) - 0.5) * self.dx

        # First derivative with periodic boundary conditions
        D = diags([-1, 0, 1], [-1, 0, 1], shape=(N, N)).toarray() / (2 * self.dx)
        D[0, :] = 0 
        D[0, 1] = 1 /(2 * self.dx)
        D[0, -1] = -1 /(2 * self.dx)

        D[-1, :] = 0
        D[-1, 0] = 1 / (2 * self.dx)
        D[-1, -2] = -1 / (2 * self.dx)

        self.D = D

        # Second derivative with periodic boundary conditions
        main_diag = -2.0 * np.ones(N)
        off_diag = np.ones(N - 1)
        Laplacian = diags([off_diag, main_diag, off_diag], [-1, 0 , 1]).toarray() / self.dx**2
        Laplacian[0, -1] = 1 / (self.dx**2)
        Laplacian[-1, 0] = 1 / (self.dx**2)

        self.Laplacian = Laplacian

    def _setup_initial_conditions(self):
        L = self.params['L']
        init_type = self.params['h_init_type']

        if init_type == 'gaussian':
            self.h_init = 0.5 + 5 * np.exp(-(self.x - L/2)**2/0.1)
        elif init_type == 'constant':
            self.h_init = np.ones_like(self.x)
        else:
            raise ValueError(f"Unknown initial condition type: {init_type}")

    @staticmethod
    def g1(a, b, c, d, k, h):
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-h/(2*c))

    @staticmethod
    def g2(a, b, c, d, k, h):
        return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-2*h/c)

    @staticmethod
    def Pi1(a, b, c, d, k, h):
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))

    @staticmethod
    def Pi2(a, b, c, d, k, h):
        return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + (2*d)/(c)*np.exp(-(2*h)/(c))
    
    def free_energy(self, h):
        """Calculates the free energy functional F[h]."""
        p = self.params
        dhdx = self.D @ h
        integrand = 0.5 * p['gamma'] * dhdx**2 + self.g1(h, p['a'], p['b'], p['c'], p['d'], p['k'])
        return np.sum(integrand) * self.dx

    def rhs(self, t, h):
        p = self.params
        h_xx = self.Laplacian @ h 
        mu = - self.Pi1(p['a'], p['b'], p['c'], p['d'], p['k'], h) - p['gamma'] * h_xx
        mu_x = self.D @ mu
        flux = self.D @ (p['Q'] * mu_x)
        source = p['g'] * h * (1 - h/ p['h_max'])

        return flux + source

    def solve(self, h0, T = 10, method = 'BDF', t_eval = None):
        if t_eval is None:
            t_eval = np.linspace(0, T, 5)
        sol = solve_ivp(self.rhs, [0, T], h0, t_eval = t_eval, method = method)
        return sol.t, sol.y


if __name__ == "__main__":

    config = OneDConfig()
    model = OneD_Thin_Film_Model(config)

    times, H = model.solve(model.h_init)

    figure_handler = fh.FigureHandler(model)
    figure_handler.plot_profiles(H, times)

    plt.show()