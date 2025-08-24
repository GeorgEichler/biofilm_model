import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import os
import csv

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model
from helper_functions import find_first_k_minima

def calculate_width_at_center(h, x, h_low, axis = 0):
    """
    Calculates the width of interval containing the center where
    the film height surpasses the first layer
    """

    H = np.asarray(h)
    x = np.asarray(x)

    # Handle 1D: just reuse the original logic
    if H.ndim == 1:
        N = H.shape[0]
        if x.shape[0] != N:
            raise ValueError("x and h must have the same length for 1D input.")
        center_idx = N // 2
        if H[center_idx] < h_low:
            return 0.0
        # expand to the right
        right_idx = center_idx
        for i in range(center_idx + 1, N):
            if H[i] >= h_low:
                right_idx = i
            else:
                break
        # expand to the left
        left_idx = center_idx
        for i in range(center_idx - 1, -1, -1):
            if H[i] >= h_low:
                left_idx = i
            else:
                break
        return float(x[right_idx] - x[left_idx])

    # 2D case: move spatial axis to axis 0 => shape (N_space, M_profiles)
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")
    if axis == 1:
        H = H.T

    N, M = H.shape
    if x.shape[0] != N:
        raise ValueError("Length of x must match the spatial dimension of h.")

    center_idx = N // 2
    widths = np.zeros(M, dtype=float)

    mask = H >= h_low  # True where h >= threshold

    for j in range(M):
        if not mask[center_idx, j]:
            widths[j] = 0.0
            continue

        # Find left boundary: nearest index < center where mask is False
        left_segment = mask[:center_idx + 1, j]              # [0 .. center]
        left_false_idxs = np.where(~left_segment)[0]
        if left_false_idxs.size == 0:
            left_idx = 0
        else:
            left_idx = int(left_false_idxs[-1] + 1)          # first True after the last False

        # Find right boundary: nearest index > center where mask is False
        right_segment = mask[center_idx:, j]                  # [center .. end]
        right_false_rel = np.where(~right_segment)[0]
        if right_false_rel.size == 0:
            right_idx = N - 1
        else:
            right_idx = int(center_idx + right_false_rel[0] - 1)  # last True before first False

        widths[j] = float(x[right_idx] - x[left_idx])

    return widths

def calculate_3_width_evolution(params, T = 1000, method = 'LSODA', 
                              init_type = 'gaussian'):
    
    model = FDM_OneD_Thin_Film_Model(**params)
    h0 = model.setup_initial_conditions(init_type)
    times, H = model.solve(h0, T = T, method = method)
    minima, _ = find_first_k_minima(3, model.f)
    h1, h2, h3 = minima
    widths_1 = calculate_width_at_center(H, model.x, h1)
    widths_2 = calculate_width_at_center(H, model.x, h2)
    widths_3 = calculate_width_at_center(H, model.x, h3)

    return times, widths_1, widths_2, widths_3

def plot_multiple_widths(params, base_params = None, T = 1000,
                         method = 'LSODA', init_type = 'gaussian',
                         plot_filename = None, csv_filename = None):
    """
    Simulation for calculating the width of the 3 first layers simultaneously
    """
    if base_params is None:
        base_params = {}


    if csv_filename:
        output_dir = os.path.dirname(csv_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fieldnames = ['t', 'First layer', 'Second layer', 'Third layer']
        with open(csv_filename,'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    _, ax = plt.subplots()

    print(f"Starting the simulation...")
    start_time = time.time()
    full_params = {**base_params, **params}
    times, widths1, widths2, widths3 = calculate_3_width_evolution(
        params=full_params,
        T=T,
        method=method,
        init_type=init_type
    )
    
    if csv_filename:
        with open(csv_filename, 'a', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for t, w1, w2, w3 in zip(times, widths1, widths2, widths3):
                writer.writerow({
                    't': t,
                    'First layer': w1,
                    'Second layer': w2,
                    'Third layer': w3
                })
            
    end_time = time.time()
    print(f"\nTotal simulation finished in {end_time - start_time:.2f}s.")

    ax.plot(times, widths1, label = '1st layer')
    ax.plot(times, widths2, label = '2nd layer')
    ax.plot(times, widths3, label = '3rd layer')

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Width of biofilm')
    #ax.set_title('Evolution of first layer')
    ax.legend()
    
    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")

    plt.show()

def calculate_width_evolution(params, T = 1000, method = 'LSODA', 
                              init_type = 'gaussian'):
    
    model = FDM_OneD_Thin_Film_Model(**params)
    h0 = model.setup_initial_conditions(init_type)
    times, H = model.solve(h0, T = T, method = method)
    widths = calculate_width_at_center(H, model.x, model.h1)


    return times, widths

def plot_widths(sweep_params, base_params = None, T = 1000,
                method = 'LSODA', init_type = 'gaussian',
                plot_filename = None, csv_filename = None):
    """
    Simulation for calculating the width of the first layer of the biofilm
    """
    if base_params is None:
        base_params = {}

    _, ax = plt.subplots()

    param_keys = list(sweep_params.keys())
    if not param_keys:
        print("Error: `sweep_params` cannot be empty!")
        return
    
    param_values = list(sweep_params.values())
    param_combinations = list(itertools.product(*param_values))
    total_sims = len(param_combinations)

    if csv_filename:
        output_dir = os.path.dirname(csv_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        fieldnames = [*param_keys, 'time', 'width']
        with open(csv_filename,'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    print(f"Starting parameter sweep with {total_sims} simulations...")
    start_time = time.time()

    for i, combo in enumerate(param_combinations):
        current_sweep_params = dict(zip(param_keys, combo))
        full_params = {**base_params, **current_sweep_params}
        
        param_str = ', '.join([f"{k}={v:.4g}" for k, v in current_sweep_params.items()])
        print(f'[{i+1}/{total_sims}] Running with {param_str}...')
        
        sim_start_time = time.time()

        times, widths = calculate_width_evolution(
            full_params, T=T, method=method, init_type=init_type
        )
        label = ", ".join([f"{k}={v:.3g}" for k, v in current_sweep_params.items()])
        ax.plot(times, widths, label = label)

        if csv_filename:
            with open(csv_filename, 'a', newline = '') as f:
                writer = csv.DictWriter(f, fieldnames=[*param_keys, 'time', 'width'])
                for t, w in zip(times, widths):
                    row = {**{k: current_sweep_params[k] for k in param_keys},
                           'time': float(t), 'width': float(w)}
                    writer.writerow(row)

        sim_end_time = time.time()
        print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")

    end_time = time.time()
    print(f"\nTotal sweep finished in {end_time - start_time:.2f}s.")


    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Width of biofilm')
    #ax.set_title('Evolution of first layer')
    ax.legend()
    
    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")

    plt.show()


if __name__ == "__main__":
    base_params = {
        'L': 200,
        'N': 2048,
        'g': 1
    }
    sweep_params = {
        'g': [1e-2]
    }

    plot_filename = "Results/plots/width_evolution_g002.png"
    csv_filename = "Results/data/width_evolution_g002.csv"

    plot_widths(
        sweep_params=sweep_params,
        T = 2500,
        base_params=base_params,
        plot_filename=plot_filename,
        csv_filename=csv_filename
    )