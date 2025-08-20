import time
import matplotlib.pyplot as plt
import itertools
from OneD_FFT_simple_model import FFT_OneD_Thin_Film_Model
from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model

def create_parameter_grid(parameter_values_dict):
    """
    Create list of parameter dictionaries from a dictionary

    Args:
        param_values_dic (dict): A dictionary with parameter name keys

    Returns:
        list: A list of dictionaries
    """
    param_names = parameter_values_dict.keys()
    values_list = parameter_values_dict.values()

    # Make Cartesian product of parameter values
    param_combination = list(itertools.product(*values_list))

    # Convert tuple into a dictionary
    param_sets = [dict(zip(param_names, combi)) for combi in param_combination]

    return param_sets

def run_sensitivity_analysis(param_sets, T = 10, initial_condition = 'gaussian',
                             const_params = {}, use_fft = False, savefig = False):
    """
    Performs a sensitivity analysis on a specified model parameter

    Args:
        param_sets (list of dict): List of parameter name and values dict
        T (float): The final simulation time
        initial_condition_type (str, optional): The initial condition to use
        const_params (dict, optional): A dictionary of other parameters to hold constant
                                       at non-default values during the analysis
    """
    print(f"--- Starting Sensitivity Analysis ---")
    start_time = time.time()
    
    fig, ax = plt.subplots(figsize = (12, 8))

    # --- Run Simulation for Each Parameter Value ---
    for i, p_set in enumerate(param_sets):
        label = ", ".join([f"{k}={v}" for k, v in p_set.items()])
        print(f" ({i+1}/{len(param_sets)}) Running simulation for: {label}")

        # Define model
        current_params = const_params.copy()
        current_params.update(p_set)
        if use_fft:
            model = FFT_OneD_Thin_Film_Model(**current_params)
        else:
            model = FDM_OneD_Thin_Film_Model(**current_params)
        
        #plot initial condition
        x_grid = model.x
        h_init = model.setup_initial_conditions(initial_condition)

        # get line to extract its color
        line, = ax.plot(x_grid, h_init, '--')
        line_color = line.get_color()

        # Solve the model
        results = model.solve(h_init, T=T, t_eval=[T])
        if use_fft:
            H = results['H']
            h_final = H[0]
        else:
            _, Y = results
            H = Y.T
            h_final = H[-1]
        ax.plot(x_grid, h_final, color = line_color, label = label, lw = 2)


    ax.set_title(f"Sensitivity Analysis at T={T}")
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Final Film Height h(x, T)')
    ax.legend(loc = 'upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    end_time = time.time()
    print(f"\nAnalysis finished in {end_time - start_time:.2f} seconds.")
    
    plt.show()

if __name__ == "__main__":
    const_params = {'amplitude': 1.5, 'g': 0.01}
    param_values = {'c': [1, 5, 10, 20]}
    param_sets = create_parameter_grid(param_values)
    run_sensitivity_analysis(
        param_sets=param_sets,
        const_params=const_params,
        T = 100
    )
    