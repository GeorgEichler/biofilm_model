import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import os
from scipy.integrate import solve_ivp

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model

def run_until_first_layer(g, base_params, T = 10000, method = 'LSODA', init_type = 'gaussian'):
    """
    Run one simulation for different growth parameters until the first layer forms
    """

    params = {**base_params, 'g': float(g)}
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

def growth_parameter_analysis(g_values, epsilon_values = [1.0], base_params = None, T = 10000,
                              method = 'LSODA', init_type = 'gaussian',
                              csv_filename = None):
    """
    Run multiple sensitivity analysis simulations for multiple parameter values
    Save output to CSV and generate log-plots
    """
    if base_params is None:
        base_params = {}
    
    fig_t, ax_t = plt.subplots()
    fig_h, ax_h = plt.subplots()

    # list of dicts for each run
    results = []

    print("Begin simulation...")
    start = time.time()

    for ei, eps in enumerate(epsilon_values):
        print(f'=== Epsilon {eps} ({ei+1}/{len(epsilon_values)}) ===')
        params = {**base_params, 'epsilon': float(eps)}
        
        t_events = np.zeros_like(g_values, dtype = float)
        h_centers = np.zeros_like(g_values, dtype = float)

        for i, g in enumerate(g_values):
            
            start_single_simulation = time.time()
            print(f'[{i+1}/{len(g_values)}] Running with g = {g:.6g}...')
            t_event, h_center = run_until_first_layer(
                g, params, T = T, method = method, init_type=init_type
            )
            t_events[i] = t_event
            h_centers[i] = h_center

            results.append({
                'epsilon': eps,
                'g': g,
                't_event': t_event,
                'h_center': h_center
            })

            if np.isnan(t_event):
                print(f' -> Event NOT reached within T = {T}; recorded as NAN')

            end_single_simulation = time.time()
            print(f"Time for simulation step: {end_single_simulation - start_single_simulation}s.")
        label = rf'$\epsilon = {eps}$'
        ax_t.plot(g_values, t_events, marker = 'o', linestyle = '-', label = label)
        ax_h.plot(g_values, h_centers, marker = 'o', linestyle = '-', label = label)


    end = time.time()
    print(f"Full simulation time: {end - start}s.")
    # Set scales and label for the plots
    ax_t.set_xscale('log')
    ax_t.set_xlabel('g')
    ax_t.set_ylabel('t_event')
    ax_t.set_title('Time to first layer')
    ax_t.legend()

    ax_h.set_xscale('log')
    ax_h.set_xlabel('g')
    ax_h.set_ylabel('h(t_event, L/2)')
    ax_h.set_title('Height at middle of profile')
    ax_h.legend()
        
    plt.show()

    if csv_filename is not None:
        os.makedirs("Results/data", exist_ok = True)

        fieldnames = ['epsilon', 'g', 't_event', 'h_center']
        with open(csv_filename, 'w', newline= '') as f:
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved simulation results to: {csv_filename}")

if __name__ == '__main__':
    #g_values = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
    g_values = [0.025, 0.05, 0.075, 0.1, 0.25]
    epsilon_values = [1, 5]

    filename = "Results/data/first_layer_g_eps.csv"

    growth_parameter_analysis(g_values=g_values, epsilon_values=epsilon_values,
                              T = 20000, csv_filename = filename)

