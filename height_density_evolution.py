import matplotlib.pyplot as plt
import numpy as np
import time
import os
import itertools

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model


def get_center_height_timeseries(params, T, t_eval, init_type = 'gaussian', method = 'LSODA'):
    """
    Runs a single thin film evolution and returns the times series of the height
    at the center of the domain x = L/2

    Args:
        params (dict): A dictionary of parameters for the model.
        T (float): The total simulation time.
        t_eval (np.ndarray): Array of time points to evaluate the solution at.
        init_type (str): The type of initial condition.
        method (str): The solver method for solve_ivp.
    Returns:
        tuple: (times, h_center_series) where `times` is the array of time points
               and `h_center_series` is the corresponding height at x=L/2.
    """

    print("-" * 50)
    param_str = ', '.join([f"{k}={v:.4g}" for k, v in params.items()])
    print(f"Running simulation with: {param_str}")
    
    # 1. Initialize the model with the given parameters
    model = FDM_OneD_Thin_Film_Model(**params)

    h_init = model.setup_initial_conditions(init_type)

    times, H = model.solve(h_init, T = T, t_eval = t_eval, method = method)

    L = model.params['L']
    center_x = L/2
    # find grid point closest to the center
    center_idx = np.argmin(np.abs(model.x - center_x))

    h_center_series = H[center_idx, :]

    return times, h_center_series


def plot_center_height_evolution(sweep_params, base_params = None, T = 1000, num_points = 101,
                                 plot_filename = None, csv_filename = None):
    """
    Perform a parameter sweep and plot the time evolution at the center height
    """

    if base_params is None:
        base_params = {}

    _, ax = plt.subplots()

    param_keys = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    param_combinations = list(itertools.product(*param_values))
    total_sims = len(param_combinations)

    print(f"Starting sweep with {total_sims} simulations...")
    start_time = time.time()

    # Define the time points for the output
    t_eval = np.linspace(0, T, num_points)

    for i, combo in enumerate(param_combinations):
        current_sweep_params = dict(zip(param_keys, combo))
        full_params = {**base_params, **current_sweep_params}


        times, h_center_series = get_center_height_timeseries(
            full_params, T, t_eval)


        label = ", ".join([f"{k}={v:.3g}" for k, v in current_sweep_params.items()])
        ax.plot(times, h_center_series, label=label)

    end_time = time.time()
    print(f"\nTotal sweep finished in {end_time - start_time:.2f}s.")


    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Center Height h(L/2, t)')
    ax.set_title('Evolution of Center Height for Different Parameters')
    ax.legend(title="Parameters")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")

    plt.show()
    
if __name__ == '__main__':

    sweep_params = {
        'g': [5e-3, 1e-2, 5e-2],
        'epsilon': [0.5, 1]
    }

    plot_filename = "Results/plots/center_height_evolution.png"

    plot_center_height_evolution(
        sweep_params=sweep_params,
        plot_filename=plot_filename
    )