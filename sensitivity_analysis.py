import numpy as np
import time
import matplotlib.pyplot as plt
from OneD_FDM_oscillatory_model import OneD_Thin_Film_Model

def run_sensitivity_analysis(param_name, param_values, T = 10, initial_condition = 'gaussian',
                             const_params = {}):
    """
    Performs a sensitivity analysis on a specified model parameter.

    Args:
        param_name (str): The name of the parameter to vary (e.g., 'Q', 'gamma')
        param_values (list or np.ndarray): A list of values to test for the parameter
        T_final (float, optional): The final simulation time. Defaults to 20
        initial_condition_type (str, optional): The initial condition to use
        const_params (dict, optional): A dictionary of other parameters to hold constant
                                       at non-default values during the analysis
    """
    print(f"--- Starting Sensitivity Analysis for Parameter: '{param_name}' ---")
    start_time = time.time()

    # Store results here
    final_profiles = {}
    
    # We need the spatial grid 'x' for plotting. We can get it from any model instance.
    # We create a temporary model, making sure to pass any constant parameters.
    temp_model = OneD_Thin_Film_Model(**const_params)
    x_grid = temp_model.x
    h_init = temp_model.setup_initial_conditions(initial_condition)

    # --- 2. Run Simulation for Each Parameter Value ---
    for p_val in param_values:
        print(f"  Running simulation for {param_name} = {p_val:.3f}...")

        # Model for the run
        current_params = const_params.copy()
        current_params[param_name] = p_val
        model = OneD_Thin_Film_Model(**current_params)

        # olve the model
        times, H = model.solve(h_init, T=T, t_eval=[T])

        # Store the final height profile
        final_profiles[p_val] = H[:, 0]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_grid, h_init, label = '$h_0$')

    for p_val, h_final in final_profiles.items():
        ax.plot(x_grid, h_final, label=f'{param_name} = {p_val:.3f}')

    ax.set_title(f"Sensitivity to '{param_name}' at T={T}")
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Final Film Height h(x, T)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    end_time = time.time()
    print(f"\nAnalysis finished in {end_time - start_time:.2f} seconds.")
    
    plt.show()

if __name__ == "__main__":
    print("\n--- Running Example 1: Analysis of 'Q' ---")
    q_values_to_test = [0.05, 0.2, 0.5, 1.0, 5.0]
    run_sensitivity_analysis(
        param_name='Q',
        param_values=q_values_to_test,
        T = 10
    )
    