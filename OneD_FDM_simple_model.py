import numpy as np
from scipy.sparse import diags
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from OneD_thin_film_model import OneD_Base_Model
from helper_functions import find_first_k_minima
from solver_wrapper import SolveIVPProgressWrapper
import figure_handler as fh
import time

class FDM_OneD_Thin_Film_Model(OneD_Base_Model):
    """
    Solve thin film equation using finite difference method and solve_ivp
    funcion for time stepping
    """

    def __init__(self, use_numba = False, **kwargs):
        
        self.use_numba = use_numba
        super().__init__(**kwargs)

    def _setup_numerical_operators(self):
        """Define sparse finite difference matrices for first and second derivative"""
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
        Event function, which triggers when the area under the curve is equal one complete
        layer on a given interval
        To investigate the mono-to-multilayer transition and critical growth rate
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
    
    # tell solve_ivp to stop once event occurs
    _event_mean_height_first_layer.terminal = True

    # trigger once event function foes from - to +
    _event_mean_height_first_layer.direction = 1 
    
    def _event_layer_transition(self, t, h):
        """
        Event function for solve_ivp triggers when the height of 2nd layer is reached
        to investigate mono-to-multilayer transition
        """

        return np.max(h) - self.h2 
    
    _event_layer_transition.terminal = True

    _event_layer_transition.direction = 1

    def rhs(self, t, h):
        """RHS for finite difference method using scipy matrices given by
        -\Delta^2 h - \Delta \epsilon \Pi(h) + G(h)
        possible to choose another non-dimensionalisation with another mobility coefficient 
        """
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
            print(f"Integration finished because the final time T={T} was reached.")

        return sol.t, sol.y


if __name__ == "__main__":
    plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })
    params = {'amplitude': 1.0, 'g': 1e-2, 'L': 100, 'N': 1024, 'epsilon': 1}
    T = 1000
    model = FDM_OneD_Thin_Film_Model(use_numba= False, **params)
    #t_eval = [500, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    t_eval = np.linspace(0, T, 6)
    
    h_init = model.setup_initial_conditions('gaussian')
    print(f"Baseline h_b ={model.h0 + 0.01}")
    times, H = model.solve(h_init, T = T, t_eval = t_eval, method = 'LSODA', event = False)
     
    
    # Calculate minima to plot the as dashed lines on the evolution plot
    h_mins, g1_mins = find_first_k_minima(
        k_minima=5, 
        f = model.f
    )

    plot_filename = 'thin_film_g10-2_multilayer_regime'
    save_filename = "Results/values/thin_film_profile.npz"
    
    figure_handler = fh.FigureHandler(model)
    #model.save_profile_values(times, H, save_filename)
    figure_handler.plot_profiles(H.T, times, pot_minima = h_mins, plot_filename = None)
    

    plt.show()