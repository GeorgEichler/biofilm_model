import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import os
from scipy.integrate import solve_ivp
import itertools

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model

def run_until_first_layer(params, T = 10000, method = 'LSODA', init_type = 'gaussian'):
    """
    Run one simulation for different growth parameters until the first layer forms
    """

    model = FDM_OneD_Thin_Film_Model(**params)

    h0 = model.setup_initial_conditions(init_type)

    sol = solve_ivp(
        fun = model.rhs,
        t_span = [0, T],
        y0 = h0,
        t_eval = [0.0],
        method = method,
        events = model._event_mean_height_first_layer
    )

    if len(sol.t_events) == 0 or sol.t_events[0].size == 0:
        print("No event occured!")
        return np.nan, np.nan
        
    t_event = float(sol.t_events[0][0])
    h_event = sol.y_events[0][0]

    L = model.params['L']
    h_center = float(np.interp(L/2, model.x, h_event))

    return t_event, h_center


def growth_parameter_analysis(sweep_params, base_params = {}, T = 10000,
                              method = 'LSODA', init_type = 'gaussian',
                              plot_filename = None, csv_filename = None):
    """
    Run multiple sensitivity analysis simulations for multiple parameter values
    Save output to CSV and generate log-plots
    Args:
        sweep_params (dict): 
            Dictionary where keys are parameter names (str) and values are lists 
            of the values to sweep. The order of keys matters for plotting.
            E.g., {'g': [0.1, 1], 'epsilon': [0.5, 1.0]}. 'g' will be the x-axis.
        base_params (dict, optional): 
            A dictionary of fixed parameters for all simulations. Defaults to {}.
        T (float): Maximum integration time.
        method (str): Integration method for solve_ivp.
        init_type (str): Type of initial condition for the model.
        plot_filename (str, optional): Path to save the plot image.
        csv_filename (str, optional): Path to save the results CSV file.
    """

    param_keys = list(sweep_params.keys())
    if not param_keys:
        print("Error: `sweep_params` cannot be empty!")
        return
    
    param_values = list(sweep_params.values())
    param_combinations = list(itertools.product(*param_values))
    total_sims = len(param_combinations)

    print(f"Starting parameter sweep with {total_sims} simulations...")
    start_time = time.time()
    
    results = []

    for i, combo in enumerate(param_combinations):
        current_sweep_params = dict(zip(param_keys, combo))
        full_params = {**base_params, **current_sweep_params}
        
        param_str = ', '.join([f"{k}={v:.4g}" for k, v in current_sweep_params.items()])
        print(f'[{i+1}/{total_sims}] Running with {param_str}...')
        
        sim_start_time = time.time()

        t_event, h_center = run_until_first_layer(
            full_params, T=T, method=method, init_type=init_type
        )
        sim_end_time = time.time()
        print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")

        result_data = {**current_sweep_params, 't_event': t_event, 'h_center': h_center}
        results.append(result_data)

    end_time = time.time()
    print(f"\nFull simulation time: {end_time - start_time:.2f}s.")

    # Save results to CSV
    if csv_filename and results:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(csv_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fieldnames = list(results[0].keys())
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved simulation results to: {csv_filename}")

    xaxis_param = param_keys[0]
    series_params = param_keys[1:] # This will be an empty list if only one param is swept

    _, ax_t = plt.subplots()
    _, ax_h = plt.subplots()

    # Group results by the series parameters to plot separate lines
    if series_params:
        grouped_results = {}
        for res in results:
            # Create a unique key for each combination of series parameter values
            series_key = tuple(res[k] for k in series_params)
            if series_key not in grouped_results:
                grouped_results[series_key] = []
            grouped_results[series_key].append(res)
        
        for series_key, series_data in grouped_results.items():
            series_data.sort(key=lambda r: r[xaxis_param])
            x_vals = [r[xaxis_param] for r in series_data]
            t_vals = [r['t_event'] for r in series_data]
            h_vals = [r['h_center'] for r in series_data]
            
            # Create a descriptive label, e.g., "epsilon=0.5, A=0.1"
            label = ", ".join([f"{key}={val}" for key, val in zip(series_params, series_key)])
            ax_t.plot(x_vals, t_vals, marker='o', linestyle='-', label=label)
            ax_h.plot(x_vals, h_vals, marker='o', linestyle='-', label=label)
    else:
        # If no series parameters, plot all data as a single line
        results.sort(key=lambda r: r[xaxis_param])
        x_vals = [r[xaxis_param] for r in results]
        t_vals = [r['t_event'] for r in results]
        h_vals = [r['h_center'] for r in results]
        ax_t.plot(x_vals, t_vals, marker='o', linestyle='-')
        ax_h.plot(x_vals, h_vals, marker='o', linestyle='-')

    # Set scales and labels for the plots
    ax_t.set_xscale('log')
    ax_t.set_xlabel(xaxis_param)
    ax_t.set_ylabel('t_event')
    ax_t.set_title('Time to first layer')
    ax_t.legend()

    ax_h.set_xscale('log')
    ax_h.set_xlabel("Growth parameter (g)")
    ax_h.set_ylabel('h(t_event, L/2)')
    #ax_h.set_title('Height at middle of profile')
    ax_h.legend()

    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        
    plt.show()

def phase_transition_analysis(params, base_params = {}, T = 10000,
                              method = 'LSODA', init_type = 'gaussian',
                              plot_filename = None, csv_filename = None):
    """
    Investigate the relationship between g and epsilon
    when arrested spreading changes to swelling
    """
    eps_values = params['epsilon']
    g_values = params['g']
    # track the current g value
    j = 0
    # result list
    result_g = []
    results = []
    start_time = time.time()

    for eps in eps_values:

        while j < len(g_values):
            g = g_values[j]
            # Update for next g value no matter what
            j += 1
            full_params = {**base_params, 'g': g, 'epsilon': eps}
            print(f"Start simulation with values epsilon={eps:.4g} and g={g:.4g}...")
            sim_start_time = time.time()
            _, h_center = run_until_first_layer(full_params,
                                                T=T, method=method, init_type=init_type)
            sim_end_time = time.time()
            print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")
            if h_center > 1.5:
                print(f"Phase transition reached for epsilon={eps:.4g}.")
                result_g.append(g)
                result_data = {'g': g, 'epsilon': eps, 'h_center': h_center}
                results.append(result_data)
                break
            print('No transition yet.')

    end_time = time.time()
    print(f"\nFull simulation time: {end_time - start_time:.2f}s.")

    # Save results to CSV
    if csv_filename and results:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(csv_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fieldnames = list(results[0].keys())
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved simulation results to: {csv_filename}")

    plt.figure()
    plt.plot(result_g, eps_values, marker='o', linestyle='-')
    plt.xlabel("Growth parameter (g)")
    plt.ylabel("Strength binding potential ($\epsilon$)")
    plt.xscale('log')
    
    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {plot_filename}")
        
    plt.show()

if __name__ == '__main__':
    # g = 0.0005 gives arrested for eps = 0.1
    # use np.logspace(np.log10(0.0005), 0)    

    choice = input("Growth parameter analysis (a) or phase transition (b)? ")

    if choice == "a":
        sweep_params = {
            'g': np.logspace(-3, -1, 51),
            'epsilon': [0.5, 1, 2]
        }

        plot_filename = "Results/plots/first_layer_many_steps_wider_interval.png"
        csv_filename = "Results/data/first_layer_many_steps_wider_interval.csv"

        growth_parameter_analysis(sweep_params=sweep_params,
                                T = 50000, plot_filename = plot_filename, csv_filename = csv_filename)
    elif choice == "b":
        params = {
            'g': np.logspace(np.log10(0.0005), -1, 101),
            'epsilon': [0.05*k for k in range(1, 41)]
        }

        plot_filename = "Results/plots/phase_transition_g_eps.png"
        csv_filename = "Results/data/phase_transition_g_eps.csv"

        phase_transition_analysis(params=params,
                                  T = 50000, plot_filename = plot_filename, csv_filename = csv_filename)

