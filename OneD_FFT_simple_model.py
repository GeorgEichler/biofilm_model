import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from OneD_thin_film_model import OneD_Base_Model
from helper_functions import find_first_k_minima
import figure_handler as fh
from tqdm import tqdm
import time

class FFT_OneD_Thin_Film_Model(OneD_Base_Model):
    """
    Class to solve the 1D thin-film equation using fast Fourier transform
    with a implicit-explicit Euler scheme
    """ 


    def _setup_numerical_operators(self):
        """Calculate the wavenumbers for the FFT"""
        p = self.params

        # wavenumbers for FFT
        k = 2*np.pi * fftfreq(p['N'], d = self.dx)
        self.k2 = k**2
        self.k4 = k**4

    
    def _time_step(self, h_hat):
        p = self.params
        dt = p['dt']

        # Inverse transform to calculate non-linear terms in real space
        h = ifft(h_hat).real

        # calculate explicit part in real space
        pi_h = self.Pi(h)
        growth_h = self.growth_term(h)

        # Fourier transform
        pi_hat = fft(pi_h)
        growth_hat = fft(growth_h)

        # get explicit part -laplacian(Pi) + growth -> k2*pi_hat + growth_hat
        N_hat = self.k2 * pi_hat + growth_hat

        # solve biharmonic term implicitely
        denom = 1 + dt * p['gamma'] * self.k4
        h_hat_new = (h_hat + dt * N_hat) / denom

        # transform back to real space
        return h_hat_new

    def solve(self, h0, T, t_eval):
        dt = self.params['dt']
        h = h0.copy()
        num_steps = int(T / dt)
        t_eval = np.asarray(t_eval)

        t_snapshots = []
        h_snapshots = []

        # Calculate indices from simulation
        raw_indices = []
        save_initial_state = False

        for t_e in t_eval:
            # check if initial state is to be recorded
            if np.isclose(t_e, 0):
                save_initial_state = True
                continue

            # Time at step i is t = (i + 1) * dt (0 indication)
            # for i + 1 be closest integer to t_e /dt take i = round(t_e/dt) - 1
            step_idx = int(np.round(t_e / dt)) - 1

            if 0 <= step_idx < num_steps:
                raw_indices.append(step_idx)
        
        # create sorted list of unique indices
        target_indices = sorted(list(set(raw_indices)))
        target_ptr = 0

        if save_initial_state:
            t_snapshots.append(0.0)
            h_snapshots.append(h.copy())

        # Use Fourier transform for for the solver
        h_hat = fft(h)

        start = time.time()
        print(f"Start integration using spectral methods in [0, {T}] with dt = {dt}...")
        for i in tqdm(range(num_steps), desc = "FFT Simulation"):
            # perform one time step
            h_hat = self._time_step(h_hat)

            # check if we want a snapshot
            if target_ptr < len(target_indices) and i == target_indices[target_ptr]:
                current_t = (i + 1) * dt
                t_snapshots.append(current_t)
                h_snapshots.append(ifft(h_hat).real.copy())
                target_ptr +=1
        end = time.time()
        print(f"Integration finished in {end - start:.3f}s.")

        results = {
            'times': np.array(t_snapshots),
            'H': np.array(h_snapshots)
        }

        return results
        
if __name__ == "__main__":
    # Simulation parameters
    T = 100
    # be careful with size of timestep for the implicit part
    t_eval = np.linspace(0, T, 5)
    params = {'amplitude': 1.5, 'g': 0.1, 'gamma': 1, 'dt': 0.0001}

    model = FFT_OneD_Thin_Film_Model(**params)
    h0 = model.setup_initial_conditions('gaussian')


    results = model.solve(h0, T, t_eval)
    times = results['times']
    H = results['H']
    
    h_mins, g_mins = find_first_k_minima(
        k_minima = 5,
        f = model.f
    )

    figure_handler = fh.FigureHandler(model)
    figure_handler.plot_profiles(H, times, pot_minima = h_mins)
    
    plt.show()