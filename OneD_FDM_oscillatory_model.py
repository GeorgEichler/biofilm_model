import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
L = 10
N = 100
dx = L/N
x = 0 + (np.arange(1, N + 1) - 0.5)*dx # cell-centred staggered grid
T = 10
num_steps = 100
dt = T/num_steps
t_eval = np.linspace(0, T, num_steps + 1)
Dif = 1.0 # diffusion coefficient
gamma = 0.1 # surface tension
h0 = 0.1 # precursor film height
g = 1    # growth term

# Initial condition
h_init = 2 * np.ones_like(x)

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

def Pi1(a, b, c, d, k, h):
    return a * np.exp(-h/c) (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/(2*c)*np.exp(-h/(2*c))

# RHS of thin film equation, assumes h to be a numpy array
def rhs_thin_film_eq(t, h):
    h_xx = Laplacian @ h
    f = -Pi1(1, 1, 1, 1, 1, h) - gamma * h_xx
    f_x =  D @ f
    flux = D @ (D * f_x)
    source = g * h * (h0 - h)
    return flux + source


# Solution of the remaining ODE
sol = solve_ivp(rhs_thin_film_eq, [0, T], h_init,  method = 'BDF', t_eval=t_eval)
H_full = sol.y.T

# Optional plotting
plt.figure(figsize=(10, 5))
plt.plot(x, H_full[0], label="t = 0")
plt.plot(x, H_full[-1], label=f"t = {T}")
plt.xlabel("x")
plt.ylabel("h(x,t)")
plt.title("Thin-film height profile")
plt.legend()
plt.grid()
plt.show()