import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
L = 10    # domain [0,L]
N = 1000   # grid numbers
dx = L/N  # space step size
x = 0 + (np.arange(1, N + 1) - 0.5)*dx # cell-centred staggered grid
T = 10          # final time
num_steps = 100 # number of time steps
t_eval = np.linspace(0, T, num_steps + 1)
Q = 0.5 # diffusion coefficient
gamma = 0.1 # surface tension
h0 = 0.21 # precursor film height
g = 0    # growth term

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

# Second derivative with Neumann boundary conditions
main_diag = -2.0 * np.ones(N)
off_diag = np.ones(N - 1)
Laplacian = diags([off_diag, main_diag, off_diag], [-1, 0 , 1]).toarray() / dx**2
Laplacian[0, 0:2] = np.array([-1, 1]) / dx**2
Laplacian[-1, -2:] = np.array([1, -1]) / dx**2

# First derivative with Neumann boundary conditions
D = diags([-1, 0, 1], [-1, 0, 1], shape=(N, N)).toarray() / (2 * dx)
D[0, 0:2] = np.array([-1, 1]) / (2 * dx) 
D[-1, -2:] = np.array([-1, 1]) / (2 * dx)  

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
    source = g * h * (h0 - h)
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