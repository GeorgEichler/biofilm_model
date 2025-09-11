import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import os
from scipy.integrate import solve_ivp
import itertools

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model


def run_until_new_layer(params, T = 10000, method = 'LSODA', init_type = 'gaussian'):
    """
    Run one simulation for different growth parameters until the second layer emerges,
    i.e. the maximum height reaches 2 and report time
    critical time = time until a second layer emerges
    """

    model = FDM_OneD_Thin_Film_Model(**params)

    h0 = model.setup_initial_conditions(init_type)

    sol = solve_ivp(
        fun = model.rhs,
        t_span = [0, T],
        y0 = h0,
        t_eval = [0.0],
        method = method,
        events = model._event_layer_transition
    )

    if len(sol.t_events) == 0 or sol.t_events[0].size == 0:
        print("No event occured!")
        return np.nan, np.nan
        
    t_critical = float(sol.t_events[0][0])
    h_event = sol.y_events[0][0]

    # check case if just boundary has been reached
    if h_event[0] > model.h1:
        t_event = np.nan
        print("Second layer has not been emerged.")

    return t_critical, h_event, model.x


def growth_parameter_analysis(sweep_params, base_params = {}, T = 10000,
                              method = 'LSODA', init_type = 'gaussian',
                              plot_filename_critical_time = None, csv_filename = None,
                              plot_filename_mean_height = None):
    """
    Run multiple sensitivity analysis simulations for multiple parameter values
    Save output to CSV and generate plots
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

    # check if parameter list is empty
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
        # define current parameter set
        full_params = {**base_params, **current_sweep_params}
        
        param_str = ', '.join([f"{k}={v:.4g}" for k, v in current_sweep_params.items()])
        print(f'[{i+1}/{total_sims}] Running with {param_str}...')
        
        sim_start_time = time.time()

        # run one simulation
        t_critical, h_critical, x = run_until_new_layer(
            full_params, T=T, method=method, init_type=init_type
        )
        # Calculate the mean height
        if np.isnan(t_critical):
            mean_height = np.nan
        else:
            L = float(x[-1] - x[0])
            mean_height = np.trapz(h_critical, x) / L
        sim_end_time = time.time()
        print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")
        if np.isnan(t_critical):
            print("The biofilm is arrested, i.e. there is no critical time.")

        result_data = {**current_sweep_params, 't_event': t_critical, 'mean_height': mean_height}
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
            
            # Create a descriptive label
            label = ", ".join([f"{key}={val}" for key, val in zip(series_params, series_key)])
            ax_t.plot(x_vals, t_vals, marker='o', linestyle='-', label=label)
    else:
        # If no series parameters, plot all data as a single line
        results.sort(key=lambda r: r[xaxis_param])
        x_vals = [r[xaxis_param] for r in results]
        t_vals = [r['t_event'] for r in results]
        ax_t.plot(x_vals, t_vals, marker='o', linestyle='-')

    # Set scales and labels for the plots
    ax_t.set_xscale('log')
    ax_t.set_xlabel("Growth parameter (g)")
    ax_t.set_ylabel('Critical time ($t_c$)')
    #ax_t.set_title('Time to second layer')
    ax_t.legend()


    if plot_filename_critical_time:
        output_dir = os.path.dirname(plot_filename_critical_time)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename_critical_time, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {plot_filename_critical_time}")

    _, ax_h = plt.subplots()

    xaxis_param = "t_event"
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
            h_vals = [r['mean_height'] for r in series_data]
            
            # Create a descriptive label, e.g., "epsilon=0.5, A=0.1"
            label = ", ".join([f"{key}={val}" for key, val in zip(series_params, series_key)])
            ax_h.plot(x_vals, h_vals, marker='o', linestyle='-', label=label)
    else:
        # If no series parameters, plot all data as a single line
        results.sort(key=lambda r: r[xaxis_param])
        x_vals = [r[xaxis_param] for r in results]
        h_vals = [r['mean_height'] for r in results]
        ax_h.plot(x_vals, h_vals, marker='o', linestyle='-')

    # Set scales and labels for the plots
    ax_h.set_xlabel(r"Critical time ($t_c$)")
    ax_h.set_ylabel(r'Mean height ($\overline{h}$)')
    #ax_t.set_title('Time to second layer')
    ax_h.legend()


    if plot_filename_mean_height:
        output_dir = os.path.dirname(plot_filename_mean_height)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename_mean_height, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {plot_filename_mean_height}")

    plt.show()

if __name__ == '__main__':
    sweep_params = {
        'g': np.logspace(-2, 0, 2),
        'epsilon': [0.5, 1, 2]
    }

    plot_filename_critical_time = "Results/plots/critical_time_long_range.png"
    csv_filename = "Results/data/critical_time_long_range.csv"
    plot_filename_mean_height = "Results/plots/critical_time_mean_height.png"

    # Choose end time to be hight enough so that a second layer always appears
    growth_parameter_analysis(sweep_params=sweep_params,
                              T = 50000, plot_filename_critical_time = None,
                              csv_filename = None,
                              plot_filename_mean_height=None)

