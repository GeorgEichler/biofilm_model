import time
import matplotlib.pyplot as plt
import itertools
from OneD_FDM_oscillatory_model import OneD_Thin_Film_Model

def create_parameter_grid(parameter_values_dic):
    """
    Create list of parameter dictionaries from a dictionary

    Args:
        param_values_dic (dict): A dictionary with parameter name keys

    Returns:
        list: A list of dictionaries
    """
    param_names = parameter_values_dic.keys()
    values_list = parameter_values_dic.values()

    # Make Cartesian product of parameter values
    param_combination = list(itertools.product(*values_list))

    # Convert tuple into a dictionary
    param_sets = [dict(zip(param_names, combi)) for combi in param_combination]

    return param_sets

def run_sensitivity_analysis(param_sets, T = 10, initial_condition = 'gaussian',
                             const_params = {}):
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

    # Store results here
    final_profiles = {}
    
    # Create temporary model to get grid
    temp_model = OneD_Thin_Film_Model(**const_params)
    x_grid = temp_model.x
    h_init = temp_model.setup_initial_conditions(initial_condition)

    # --- Run Simulation for Each Parameter Value ---
    for i, p_set in enumerate(param_sets):
        label = ", ".join([f"{k}={v}" for k, v in p_set.items()])
        print(f" ({i+1}/{len(param_sets)}) Running simulation for: {label}")

        # Define model
        current_params = const_params.copy()
        current_params.update(p_set)
        model = OneD_Thin_Film_Model(**current_params)

        # Solve the model
        _, H = model.solve(h_init, T=T, t_eval=[T])

        # Store the final height profile
        final_profiles[label] = H[:, 0]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_grid, h_init, label = '$h_0$')

    for label, h_final in final_profiles.items():
        ax.plot(x_grid, h_final, label=label, lw = 2) # lw adjusts the line width

    ax.set_title(f"Sensitivity Analysis at T={T}")
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Final Film Height h(x, T)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    end_time = time.time()
    print(f"\nAnalysis finished in {end_time - start_time:.2f} seconds.")
    
    plt.show()

if __name__ == "__main__":
    d_values = {'d': [-0.02, 0, 0.02]}
    param_sets = create_parameter_grid(d_values)
    run_sensitivity_analysis(
        param_sets=param_sets,
        T = 10
    )
    