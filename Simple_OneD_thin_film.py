import fenics as fe
import matplotlib.pyplot as plt
import config as cfg
import figure_handler as fh

# --- Configuration ---
config = cfg.BaseModelConfig(
    nx=501,
    domain_length=100,
    num_steps=100,
    final_time=50,
    Q=0.01
)
figure_handler = fh.FigureHandler(config)

# Function Spaces for Mixed System
V = fe.FunctionSpace(config.mesh, "Lagrange", 1) # Space for h
W = fe.FunctionSpace(config.mesh, "Lagrange", 1) # Space for w
element = fe.MixedElement([V.ufl_element(), W.ufl_element()])
VW = fe.FunctionSpace(config.mesh, element) # Function space where elements have a pair of values at each nodes

# Initial Conditions
h0_expr = fe.Expression("1.0 + 0.1*sin(2*pi*x[0]/L)", L=config.L, degree=2)
h0 = fe.interpolate(h0_expr, V)
h0 = config.set_ics('constant')

# 
w_trial = fe.TrialFunction(W)
vw_test = fe.TestFunction(W)
a_w = w_trial * vw_test * fe.dx
L_w = -fe.dot(fe.grad(h0), fe.grad(vw_test)) * fe.dx
w0 = fe.Function(W)
fe.solve(a_w == L_w, w0) # Relate w0 to h0 by w = \Delta h

# Set up functions for time-stepping
# H_k holds the solution (h, w) from the previous step
H_k = fe.Function(VW)
# Use fe.assign to project the separate h0 and w0 into the mixed space H_k
fe.assign(H_k, [fe.interpolate(h0, V), fe.interpolate(w0, W)])

# H is the function we solve for at the current step
H = fe.Function(VW)

# --- Variational Problem Definition ---
(h, w) = fe.split(H) # split the coordinates by projecting to the according coordinates
(h_k_split, w_k_split) = fe.split(H_k) # Use the components of H_k in the form
(vh, vw) = fe.TestFunctions(VW)

# Chemical potential
mu = -1/h**6 + 1/h**3 + w

# Weak form of the PDE for h (F1)
F1 = ((h - h_k_split)/config.dt)*vh*fe.dx \
     + fe.dot(h**3 * fe.grad(mu), fe.grad(vh)) * fe.dx \
     + config.Q * mu * vh * fe.dx

# Weak form of the auxiliary equation: w = -∇²h (F2)
F2 = w*vw*fe.dx - fe.dot(fe.grad(h), fe.grad(vw)) * fe.dx

# Total variational form and its Jacobian
F = F1 + F2
J = fe.derivative(F, H)

# Solver Setup
problem = fe.NonlinearVariationalProblem(F, H, bcs=[], J=J)
solver = fe.NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["relaxation_parameter"] = 0.9
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["absolute_tolerance"] = 1e-9
prm["newton_solver"]["maximum_iterations"] = 50

# Time Stepping Loop
# Store initial h for plotting
iterates = [h0.copy(deepcopy=True)]
t = 0.0

for i in range(config.num_steps):
    t += config.dt
    print(f"Solving step {i+1}/{config.num_steps} at time t={t:.2f}")

    # Set the initial guess for the Newton solver to be the solution from the previous step.
    H.assign(H_k)

    solver.solve()

    # Update H_k for the next time step
    H_k.assign(H)

    # Store a copy of the h component for plotting
    h_sol, w_sol = H.split(deepcopy=True) # deepcopy is fine here for splitting into new functions
    iterates.append(h_sol)

# Plotting
timestamps = [0, config.num_steps // 2, config.num_steps]
figure_handler.height_profile(iterates, timestamps, savefig=True)
plt.show()
