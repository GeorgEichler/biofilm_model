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


"""
# Parameters
L = 10    # domain [0,L]
N = 1000   # grid numbers
dx = L/N  # space step size
x = 0 + (np.arange(1, N + 1) - 0.5)*dx # cell-centred staggered grid
T = 100          # final time
num_steps = 100 # number of time steps
t_eval = np.linspace(0, T, 5) 
Q = 0.5 # diffusion coefficient
gamma = 0.1 # surface tension
h_max = 5 # maximal film height
g = 0.01    # growth term

# Parameters for binding potential
a = 0.1
b = np.pi/2
c = 1
d = 0
k = 2*np.pi

# Initial conditions
h_init = np.ones_like(x)
h_init = 0.5 + 5 * np.exp(-(x - L/2)**2/0.1)
#h_init = 0.1 - 0.001 * (x - L/2)**2

# First derivative with periodic boundary conditions
D = diags([-1, 0, 1], [-1, 0, 1], shape=(N, N)).toarray() / (2 * dx)
D[0, :] = 0 
D[0, 1] = 1 /(2 * dx)
D[0, -1] = -1 /(2 * dx)

D[-1, :] = 0
D[-1, 0] = 1 / (2 * dx)
D[-1, -2] = -1 / (2 * dx)

# Second derivative with periodic boundary conditions
main_diag = -2.0 * np.ones(N)
off_diag = np.ones(N - 1)
Laplacian = diags([off_diag, main_diag, off_diag], [-1, 0 , 1]).toarray() / dx**2
Laplacian[0, -1] = 1 / (dx**2)
Laplacian[-1, 0] = 1 / (dx**2)


def g1(a, b, c, d, k, h):
    return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-h/(2*c))

def g2(a, b, c, d, k, h):
    return a * np.cos(h * k + b) * np.exp(-h/c) + d * np.exp(-2*h/c)


def Pi1(a, b, c, d, k, h):
    return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))

def Pi2(a, b, c, d, k, h):
    return a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + (2*d)/(c)*np.exp(-(2*h)/(c))

# RHS of thin film equation, assumes h to be a numpy array
def rhs_thin_film_eq(t, h):
    h_xx = Laplacian @ h
    f = -Pi1(a, b, c, d, k, h) - gamma * h_xx
    f_x =  D @ f
    flux = D @ (Q * f_x)
    source = g * h * (1 - h/h_max)
    return flux + source

def free_energy(h):
    dhdx = D @ h
    integrand = 0.5 * gamma * dhdx**2 + g1(a, b, c, d, k, h)

    return np.sum(integrand) * dx

# Solution of the remaining ODE
sol = solve_ivp(rhs_thin_film_eq, [0, T], h_init,  method = 'BDF', t_eval=t_eval)
times = sol.t
H  = sol.y
F_values = np.array([free_energy(H[:,i]) for i in range(len(times))])

plt.figure()
plt.plot(times, F_values, '-o')
plt.xlabel('t')
plt.ylabel('F[h(t)]')
plt.title('Free energy evolution')
plt.grid(True)

# plot binding potential
h_array = np.linspace(0, 10, 1001)
plt.figure()
plt.plot(h_array, g1(a, b, c, d, k, h_array))
plt.xlabel('h')
plt.ylabel('g(h)')
plt.title("Binding potential")

# Plot solution
n_plots = 5
idxs    = np.linspace(0, len(times)-1, n_plots, dtype=int)

plt.figure()
for i in idxs:
    plt.plot(x, H[:, i], label=f"t = {times[i]:.2f}")
plt.xlabel("x")
plt.ylabel("h(x,t)")
plt.title("Thin-film height profile")
plt.legend()
plt.grid()
plt.show()
"""


config = OneDConfig()
model = OneD_Thin_Film_Model(config)

times, H = model.solve(model.h_init)

figure_handler = fh.FigureHandler()
figure_handler.plot_profiles(model.x, H, times)
plt.show()