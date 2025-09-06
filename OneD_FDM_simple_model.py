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
def _rhs_stencil_numba(h, dx, N, epsilon, g, h_max, h0, h_a, a, b, c, d, e, k):
    """RHS calculation using direct stencils, optimized with Numba."""
    
    flux = np.empty_like(h)
    dx2 = dx * dx

    # --- Main loop with stencil fusion ---
    for i in range(N):
        # Get indices for a 5-point stencil window (for the ∇⁴ operator)
        i_p2 = (i + 2) % N
        i_p1 = (i + 1) % N
        i_m1 = (i - 1 + N) % N
        i_m2 = (i - 2 + N) % N

        # --- On-the-fly calculation of mu for points i-1, i, i+1 ---
        
        # mu at point i-1
        h_xx_im1 = (h[i] - 2 * h[i_m1] + h[i_m2]) / dx2
        pi_im1 = a * np.exp(-h[i_m1]/c) * (k * np.sin(h[i_m1]*k+b) + 1/c*np.cos(h[i_m1]*k+b)) + d/e*np.exp(-h[i_m1]/e)
        mu_im1 = -epsilon * pi_im1 - h_xx_im1

        # mu at point i
        h_xx_i = (h[i_p1] - 2 * h[i] + h[i_m1]) / dx2
        pi_i = a * np.exp(-h[i]/c) * (k * np.sin(h[i]*k+b) + 1/c*np.cos(h[i]*k+b)) + d/e*np.exp(-h[i]/e)
        mu_i = -epsilon * pi_i - h_xx_i

        # mu at point i+1
        h_xx_ip1 = (h[i_p2] - 2 * h[i_p1] + h[i]) / dx2
        pi_ip1 = a * np.exp(-h[i_p1]/c) * (k * np.sin(h[i_p1]*k+b) + 1/c*np.cos(h[i_p1]*k+b)) + d/e*np.exp(-h[i_p1]/e)
        mu_ip1 = -epsilon * pi_ip1 - h_xx_ip1
        
        # --- Calculate flux at point i using the on-the-fly mu values ---
        flux[i] = (mu_ip1 - 2 * mu_i + mu_im1) / dx2

    # Source term is calculated in a separate loop as it's purely local
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
        x = self.x
        L = self.params['L']
        #set the boundaries of the interval [x0,x1] where the mean will be computed
        x0 = 0.05*L
        x1 = 0.95*L

        # Indices of the boundary interval points [x0, x1]
        i0 = np.searchsorted(x, x0, side = 'left')
        i1 = np.searchsorted(x, x1, side = 'right')

        # Correct the precursor height and use trapezoidal rule for integration 
        h_no_precursor = h[i0:i1] - self.h0
        mean_h = np.trapz(h_no_precursor, x[i0:i1])/(x1-x0)

        # Event occurs when mean height equal to the height of first layer
        return mean_h - (self.h1 - self.h0)
    
    # We need to tell the solver to stop when this event occurs.
    # We do this by setting an attribute on the function object itself.
    _event_mean_height_first_layer.terminal = True

    # We also want the event to trigger only when mean(h) is increasing through 1.
    # This prevents it from triggering if the mean somehow starts above 1 and decreases.
    _event_mean_height_first_layer.direction = 1 # Trigger when event function goes from - to +
    
    def _event_layer_transition(self, t, h):
        """
        Event function for solve_ivp triggers when the height of 2nd layer is reached
        """

        return np.max(h) - self.h2 
    
    _event_layer_transition.terminal = True

    _event_layer_transition.direction = 1

    def _rhs_scipy(self, t, h):
        """RHS for finite difference method using scipy matrices"""
        p = self.params
        a = p['a']; b = p['b']; c = p['c']; d = p['d']; e = p['e']; k = p['k']
        Pi_h = a * np.exp(-h/c) * (k * np.sin(h * k + b) + 1/c * np.cos(h * k + b)) + d/e*np.exp(-h/e)

        growth = p['g'] * (h - self.ha) * (1 - h/p['h_max']) * (1 - np.exp( (p['hf'] - h) ))


        #flux = self.Laplacian @ (-p['epsilon'] * Pi_h - self.Laplacian @ h)

        h_xx = self.Laplacian @ h 
        mu = - self.params['epsilon'] * Pi_h - h_xx
        flux = self.Laplacian @ mu
        #mu_x = self.D @ mu
        #flux = self.D @ (h**3 * mu_x)
        return flux + growth


    # Right hand side of PDE
    def rhs(self, t, h):
        if self.use_numba:
            p = self.params
            # Numba function is called with parameters unpacked from the dict
            return _rhs_stencil_numba(h, self.dx, p['N'], p['epsilon'], 
                                  p['g'], p['h_max'], self.h0, self.ha, p['a'], p['b'], 
                                  p['c'], p['d'], p['e'], p['k'])
        else:
            return self._rhs_scipy(t, h)
        
    # Good possible methods due to the stiffness are LSODA, BDF or Radau
    def solve(self, h0, T = 10, method = 'LSODA', t_eval = None, event = None):
        start = time.time()
        print(f"Start integration using finite differences and {method} method in [0, {T}]...")

        rhs_to_use = SolveIVPProgressWrapper(self.rhs, T, report_step_percent=1)
        if event == 'mean_first:layer':
            sol = solve_ivp(
                rhs_to_use,
                [0, T], 
                h0, 
                t_eval = t_eval, 
                method = method,
                events=self._event_mean_height_first_layer
                )
        elif event == 'layer_transition':
            sol = solve_ivp(
                rhs_to_use,
                [0, T],
                h0,
                t_eval = t_eval,
                method = method,
                event = self._event_layer_transition
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


if __name__ == "__main__":
    
    params = {'amplitude': 1.0, 'g': 1, 'L': 200, 'N': 2048, 'epsilon': 1}
    T = 2500
    model = FDM_OneD_Thin_Film_Model(use_numba= False, **params)
    #t_eval = [500, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    t_eval = np.linspace(0, T, 6)
    """
    h_init = model.setup_initial_conditions('gaussian')
    print(f"Baseline h_b ={model.h0 + 0.01}")
    times, H = model.solve(h_init, T = T, t_eval = t_eval, method = 'LSODA', event = False)
    
    #h_final = H[:, -1]
    #model.params['g'] = 0
    #times, H = model.solve(h_final, T, t_eval = t_eval)    
    
    h_mins, g1_mins = find_first_k_minima(
        k_minima=5, 
        f = model.f
    )

    plot_filename = 'thin_film_g10-1_eps1_t200_L200'
    save_filename = "Results/values/thin_film_profile.npz"
    """
    figure_handler = fh.FigureHandler(model)
    #model.save_profile_values(times, H, save_filename)
    #figure_handler.plot_profiles(H.T, times, pot_minima = h_mins, plot_filename = None)
    
    #figure_handler.plot_binding_energy(model.f, filename = "binding_potential")
    figure_handler.plot_growth_function(model.growth_term, filename = "growth_function")
    #print(f"Minima of g\u2081 are found at {h_mins} \n with values {g1_mins}.")
    #figure_handler.plot_free_energy(H, times)

    plt.show()