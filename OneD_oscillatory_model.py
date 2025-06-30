import fenics as fe
import matplotlib.pyplot as plt
import config as cfg
import figure_handler as fh

# --- Configuration ---
config = cfg.OscillatoryModelConfig(
    nx=501,
    domain_length=100,
    num_steps=1000,
    final_time=10,
    mobility_coefficient=1,
    surface_tension=1,
    growth_rate=1
)
a = fe.Constant(1.0)
b = fe.Constant(1.0)
c = fe.Constant(1.0)
d = fe.Constant(1.0)
k = fe.Constant(1.0)
figure_handler = fh.FigureHandler(config)

# Function Spaces for Mixed System
P1 = fe.FiniteElement('P', config.mesh.ufl_cell(), 1)
W_elem = fe.MixedElement([P1, P1])
W = fe.FunctionSpace(config.mesh, W_elem)

# Define test and solution functions
(v, q) = fe.TestFunction(W)
w = fe.Function(W)
(h, mu) = fe.split(w)

# previous step
w_n = fe.Function(W)
(h_n, mu_n) = fe.split(w_n)

# 1. Get the individual component function spaces from the mixed space W.
V_h = W.sub(0).collapse()  # Space for h
V_mu = W.sub(1).collapse() # Space for mu


h_init = config.set_ics("constant", V_h)

mu_init = fe.interpolate(fe.Constant(0.0), V_mu)

# Creating a FunctionAssigner.
# This object learns how to transfer data from the sub-spaces to the mixed space.
assigner = fe.FunctionAssigner(W, [V_h, V_mu])

# Using assigner to populate the mixed function w_n.

assigner.assign(w_n, [h_init, mu_init])

# Disjoining pressure
Pi = config.disjoining_pressure1(a, b, c, d, k, h)

# Define weak formulations
F1 = (h - h_n)/config.dt *v *fe.dx - config.D * fe.dot(fe.grad(mu),fe.grad(v)) *fe.dx - config.g * h*(config.h0 - h) * v * fe.dx
F2 = 1 * q * fe.dx - config.gamma * fe.dot(fe.grad(h), fe.grad(q)) * fe.dx + Pi * q * fe.dx

# Total variation and Jacobian
F = F1 + F2
J = fe.derivative(F, w)

# Solver Setup, need to give problem on mixed function space
problem = fe.NonlinearVariationalProblem(F, w, bcs=[], J=J)
solver = fe.NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["linear_solver"] = "mumps" # A good, robust direct solver
prm["newton_solver"]["error_on_nonconvergence"] = False # Don't crash immediately
prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True


# Time Stepping Loop
# Store initial h for plotting
h_init, _ = w_n.split(deepcopy=True)
iterates = [h_init]
t = 0.0

for i in range(config.num_steps):
    t += config.dt
    print(f"Solving step {i+1}/{config.num_steps} at time t={t:.2f}")

    # Set the initial guess for the Newton solver to be the solution from the previous step.
    w.assign(w_n)

    solver.solve()

    # Update H_k for the next time step
    w_n.assign(w)

    # Store a copy of the h component for plotting
    h_sol, _ = w.split(deepcopy=True) # deepcopy is fine here for splitting into new functions
    iterates.append(h_sol)

# Plotting
timestamps = [0, config.num_steps // 2, config.num_steps]
figure_handler.height_profile(iterates, timestamps, savefig=True)
plt.show()