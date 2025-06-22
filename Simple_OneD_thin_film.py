import fenics as fe
import ufl #needed to use exp, tanh etc. function for fenics code
import config as cfg

def f(h):
    return - 1/h**6 + 1/h**3 - fe.div(fe.grad(h))

config = cfg.BaseModelConfig()

#Initial condition
h_k = config.set_ics()
h_0 = h_k.copy(deepcopy = True)

#Weak formulation
v = fe.TestFunction(config.V)
h = fe.Function(config.V)

# Thin-film equation
fh = f(h) #is treated as a symbolic expression afterwards 
F = ( (h - h_k) / config.dt ) * v * fe.dx \
    + fe.inner(h**3 * fe.grad(fh), fe.grad(v)) * fe.dx \
    + config.Q * fh * v * fe.dx

J = fe.derivative(F, h)

# Set up solver
problem = fe.NonlinearVariationalProblem(F, h, J = J)
solver = fe.NonlinearVariationalSolver(problem)

iterates = [h_0]

for i in range(config.num_steps):
    solver.solve()
    h_k.assign(h)
    iterates.append(h_k.copy(deepcopy=True))
