import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from OneD_thin_film_model import OneD_Base_Model
from helper_functions import find_first_k_minima
from solver_wrapper import SolveIVPProgressWrapper
import figure_handler as fh
import time
from numba import njit

# Cannot use class methods or dictionaries for njitted functions
@njit
def _rhs_stencil_numba(h, dx, N, gamma, epsilon, g, h_max, h0, h_a, a, b, c, d, e, k):
    """RHS calculation using direct stencils, optimized with Numba."""
    
    # Allocate arrays for intermediate results
    h_xx = np.empty_like(h)
    mu = np.empty_like(h)
    flux = np.empty_like(h)
    
    # --- Pi1 calculation (same as before) ---
    exp_h_c = np.exp(-h / c)
    cos_term = np.cos(h * k + b)
    sin_term = np.sin(h * k + b)
    pi1 = a * exp_h_c * (k * sin_term + 1/c * cos_term) + d/(e)*np.exp(-h/(e))

    # --- Derivative calculations using stencils in a loop ---
    dx2 = dx * dx

    for i in range(N):
        # Periodic boundary conditions using modulo
        i_plus_1 = (i + 1) % N
        i_minus_1 = (i - 1 + N) % N
        
        # Second derivative of h
        h_xx[i] = (h[i_plus_1] - 2 * h[i] + h[i_minus_1]) / dx2
    
    mu = -epsilon * pi1 - gamma * h_xx
    
        
    for i in range(N):
        i_plus_1 = (i + 1) % N
        i_minus_1 = (i - 1 + N) % N
        
        # First derivative of mu_x (this is the flux term)
        flux[i] = (mu[i_plus_1] - 2 * mu[i] + mu[i_minus_1]) / dx2

    source = np.empty_like(h)
    for i in range(N):
        hi = h[i]
        source[i] = g * (1.0 - hi / h_max) * (hi - h_a) * (1.0 - np.exp(h0 - hi))

    return flux + source

class FDM_OneD_Thin_Film_Model(OneD_Base_Model):
    """
    Solve thin film equation using finite difference method
    """

    def __init__(self, use_numba = False, **kwargs):
        
        self.use_numba = use_numba
        super().__init__(**kwargs)

    def _setup_numerical_operators(self):
        """Define sparse finite difference matrices"""
        N = self.params['N']

        # First derivative with periodic boundary conditions, also needed for free energy
        D = diags(diagonals=[-1,1], offsets=[-1,1], shape=(N, N), format= 'lil')
        D[0, -1] = -1
        D[-1, 0] = 1
        self.D = (D / (2 * self.dx)).asformat('csr')
        
        # Second derivative with periodic boundary conditions
        Laplacian = diags(diagonals=[1,-2,1], offsets=[-1,0,1], shape = (N, N), format = 'lil')
        Laplacian[0, -1] = 1
        Laplacian[-1, 0] = 1
        self.Laplacian = (Laplacian / (self.dx**2)).asformat('csr')
    
    def free_energy(self, h):
        """Calculates the free energy functional F[h]
        Args:
            h (np.ndarray): array of height thickness
        Returns:
            Contribution of surface and potential energy
        """
        p = self.params
        dhdx = self.D @ h
        surface_energy_density = 0.5 * dhdx**2
        potential = p['epsilon'] * self.f(h)
        return [np.sum(surface_energy_density) * self.dx, np.sum(potential) * self.dx]
    
    def _event_mean_height_first_layer(self, t, h):
        """
        Event function for solve_ivp
        Triggers when mean height of thin film reaches first layer
        """

        return np.mean(h) - self.h1
    
    # We need to tell the solver to stop when this event occurs.
    # We do this by setting an attribute on the function object itself.
    _event_mean_height_first_layer.terminal = True

    # We also want the event to trigger only when mean(h) is increasing through 1.
    # This prevents it from triggering if the mean somehow starts above 1 and decreases.
    _event_mean_height_first_layer.direction = 1 # Trigger when event function goes from - to +
    

    def _rhs_scipy(self, t, h):
        """RHS for finite difference method using scipy matrices"""
        h_xx = self.Laplacian @ h 
        mu = - self.params['epsilon'] * self.Pi(h) - h_xx
        flux = self.Laplacian @ mu
        #mu_x = self.D @ mu
        #flux = self.D @ (h**3 * mu_x)
        return flux + self.growth_term(h)


    # Right hand side of PDE
    def rhs(self, t, h):
        if self.use_numba:
            p = self.params
            # Numba function is called with parameters unpacked from the dict
            return _rhs_stencil_numba(h, self.dx, p['N'], p['gamma'], p['epsilon'], 
                                  p['g'], p['h_max'], self.h0, self.ha, p['a'], p['b'], 
                                  p['c'], p['d'], p['e'], p['k'])
        else:
            return self._rhs_scipy(t, h)
        
    # Good possible methods due to the stiffness are LSODA, BDF or Radau
    def solve(self, h0, T = 10, method = 'LSODA', t_eval = None, event = False):
        start = time.time()
        print(f"Start integration using finite differences and {method} method in [0, {T}]...")
        if t_eval is None:
            t_eval = np.linspace(0, T, 5)

        rhs_to_use = SolveIVPProgressWrapper(self.rhs, T, report_step_percent=5)
        if event:
            sol = solve_ivp(
                rhs_to_use,
                [0, T], 
                h0, 
                t_eval = t_eval, 
                method = method,
                events=self._event_mean_height_first_layer
                )
        else:
            sol = solve_ivp(rhs_to_use, [0, T], h0, t_eval=t_eval, method = method)
        end = time.time()
        print(f"\nIntegration finished in {end - start:.3f}s.")

        # Check if the simulation was terminated by the event
        if sol.status == 1:
            print(f"Event triggered: Mean first layer reached at t = {sol.t_events[0][0]:.4f}")
        elif sol.status == 0:
            print("Integration finished because the end time T was reached.")

        return sol.t, sol.y
    
    def calculate_contact_angles(self, h, h_contact_threshold = None):
        """
        Compute the contact andgle of a given height profile

        Parameter:
        h (np.ndarray): 1D array representing the height profile
        h_contact_threshold (float): height at which to define contact line

        Returns
        """


if __name__ == "__main__":
    params = {'amplitude': 1.0, 'g': 10**(-2)}
    T = 1000
    model = FDM_OneD_Thin_Film_Model(use_numba= False, **params)
    t_eval = np.linspace(0, T, 5)
    t_plot = np.linspace(0, T, 5)

    h_init = model.setup_initial_conditions('gaussian')
    times, H = model.solve(h_init, T = T, t_eval = t_eval, method = 'LSODA')

    model.save_profile_values(times, H, "Results/values/thinfilm_profiles.npz")

    
    h_mins, g1_mins = find_first_k_minima(
        k_minima=5, 
        f = model.f
    )
    figure_handler = fh.FigureHandler(model)
    figure_handler.plot_profiles(H.T, times, pot_minima = h_mins, filename = 'thin_film_g01')
    #figure_handler.plot_growth(H.T, times)
    #figure_handler.plot_binding_energy(model.f)
    #print(f"Minima of g\u2081 are found at {h_mins} \n with values {g1_mins}.")
    #figure_handler.plot_free_energy(H, times)

    plt.show()