import fenics as fe
#import ufl
#import numpy as np may needed later

class BaseModelConfig:
    def __init__(self, grid = 100, domain_length = 25, num_steps = 50, final_time = 10, Q = 1):

        self.nx = grid
        self.L = domain_length
        self.num_steps = num_steps
        self.T = final_time
        self.dt = self.T / self.num_steps
        self.Q = Q
        self.mesh = fe.IntervalMesh(self.nx, 0, self.L)
        self.V = fe.FunctionSpace(self.mesh, "Lagrange", 1)

    def set_ics(self):
        h_init = fe.interpolate(fe.Expression("10* exp(-pow( (x[0] - L/2)/10 , 2))", L = self.L, degree = 2), self.V)

        return h_init
