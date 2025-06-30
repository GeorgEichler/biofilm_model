import fenics as fe
#import ufl
#import numpy as np may needed later

class JonesPotentialModelConfig:
    def __init__(self, nx = 1001, domain_length = 50, num_steps = 50, final_time = 10, Q = 1):

        self.nx = nx                     # division of interval
        self.L = domain_length             # interval length [0,L]
        self.num_steps = num_steps         # number of time steps
        self.T = final_time                # time interval [0,T]
        self.dt = self.T / self.num_steps  # time step size
        self.Q = Q                         # osmotic mobility coefficient
        self.mesh = fe.IntervalMesh(self.nx, 0, self.L)      # definition of mesh
        self.V = fe.FunctionSpace(self.mesh, "Lagrange", 1)  # definition of function space

        self.h_options = {
            "constant": fe.Constant(1),
            "gaussian": fe.Expression("10* exp(-pow( (x[0] - L/2)/10 , 2))", L = self.L, degree = 2)
        }

    def set_ics(self, h_option):
        h_init = fe.interpolate(self.h_options[h_option], self.V)

        return h_init

