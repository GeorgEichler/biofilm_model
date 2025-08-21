import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import os

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model

def calculate_width_at_center(h, x, h1, axis = 0):
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
        if H[center_idx] < h1:
            return 0.0
        # expand to the right
        right_idx = center_idx
        for i in range(center_idx + 1, N):
            if H[i] >= h1:
                right_idx = i
            else:
                break
        # expand to the left
        left_idx = center_idx
        for i in range(center_idx - 1, -1, -1):
            if H[i] >= h1:
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

    mask = H >= h1  # True where h >= threshold

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

def calculate_width_evolution(params, T = 1000, method = 'LSODA', 
                              init_type = 'gaussian'):
    
    model = FDM_OneD_Thin_Film_Model(**params)
    h0 = model.setup_initial_conditions(init_type)
    times, H = model.solve(h0, T = T, method = method)
    widths = calculate_width_at_center(H, model.x, model.h1)

    return times, widths

def plot_widths(sweep_params, base_params = None, T = 1000,
                method = 'LSODA', init_type = 'gaussian',
                plot_filename = None):
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
        sim_end_time = time.time()
        print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")

    end_time = time.time()
    print(f"\nTotal sweep finished in {end_time - start_time:.2f}s.")


    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Width of biofilm')
    ax.set_title('Evolution of first layer')
    ax.legend(title="Parameters")
    
    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")

    plt.show()


if __name__ == "__main__":
    sweep_params = {
        'g': [1e-2, 1e-1, 1]
    }

    plot_filename = "Results/plots/width_evolution.png"

    plot_widths(
        sweep_params=sweep_params,
        plot_filename=plot_filename
    )