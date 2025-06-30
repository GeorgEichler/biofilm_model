import fenics as fe
import ufl
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

class OscillatoryModelConfig:
    def __init__(self, nx = 1001, domain_length = 100, num_steps = 50, final_time = 10,
                 mobility_coefficient = 1, surface_tension = 1, growth_rate = 1):
        self.nx = nx
        self.L = domain_length
        self.num_steps = num_steps
        self.T = final_time
        self.D = mobility_coefficient
        self.gamma = surface_tension
        self.g = growth_rate
        self.dt = self.T/self.num_steps

        self.mesh = fe.IntervalMesh(self.nx, 0, self.L)
        self.V = fe.FunctionSpace(self.mesh, "Lagrange", 1)

        self.h_options = {
            "constant": fe.Constant(1)
        }

    def set_ics(self, h_option = None, h_init = None):

        if isinstance(h_option, str):
            try:
                expr = self.h_options[h_option]
            except KeyError:
                raise ValueError(f"Unknown h_option '{h_option}'")
            
            else:
                expr = h_option

            return fe.interpolate(expr, self.V)

        h_init = fe.interpolate(self.h_options[h_option], self.V)

        return h_init
    
    # binding potential g_1
    # g_1(h) &= a \cos(hk + b)e^{-h/c} + d e^{-h/(2c)}
    def disjoining_pressure1(self, a, b, c, d, k, h):
       a = ufl.Constant(a)
       b = ufl.Constant(b)
       c = ufl.Constant(c)
       d = ufl.Constant(d)
       k = ufl.Constant(k)
       return  a * ufl.exp(-h/c) * (k * ufl.sin(k*h + b) + 1/c * ufl.cos(k * h + b)) \
               + d/(2*c) * ufl.exp(-h/(2*c))
    
    # binding potential g_2
    # g_2(h) &= a \cos(hk + b)e^{-h/c} + d e^{-(2h)/c}
    def disjoining_pressure2(self, a, b, c, d, k, h):
       a = ufl.Constant(a)
       b = ufl.Constant(b)
       c = ufl.Constant(c)
       d = ufl.Constant(d)
       k = ufl.Constant(k)
       return  a * ufl.exp(-h/c) * (k * ufl.sin(k*h + b) + 1/c * ufl.cos(k * h + b)) \
               + (2*d)/(c) * ufl.exp(-(2*h)/c)