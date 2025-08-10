import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model

def run_until_first_layer(g, base_params, T = 10000, method = 'LSODA', init_type = 'gaussian'):
    """
    Run simulation for different growth parameters until the first layer forms
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

def growth_parameter_analysis(g_values, base_params = None, T = 10000,
                              method = 'LSODA', init_type = 'gaussian'):
    if base_params is None:
        base_params = {}
    
    t_events = np.zeros_like(g_values, dtype = float)
    h_centers = np.zeros_like(g_values, dtype = float)

    for i, g in enumerate(g_values):
        print(f'[{i+1}/{len(g_values)}] Running with g = {g:.6g}...')
        t_event, h_center = run_until_first_layer(
            g, base_params, T = T, method = method, init_type=init_type
        )
        t_events[i] = t_event
        h_centers[i] = h_center
        if np.isnan(t_event):
            print(' -> Event NOT reached within T = {T}; recorded as NAN')

        
    plt.figure()
    plt.plot(g_values, t_events, marker = 'o-')
    plt.xscale('log')
    plt.xlabel('g')
    plt.ylabel('t_event')
    plt.title('Time to first layer')

    plt.figure()
    plt.plot(g_values, h_centers, marker = 'o-')
    plt.xscale('log')
    plt.xlabel('g')
    plt.ylabel('h(t_event, L/2)')
    plt.title('Height of middle profile')
    plt.show()

if __name__ == '__main__':
    g_values = [0.005, 0.01, 0.05, 0.1, 0.5, 1]

    growth_parameter_analysis(g_values=g_values, T = 20000)

