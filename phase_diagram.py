import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from layer_formation_analysis import run_until_first_layer

def get_g_c_from_epsilon(csv_filename, epsilon, eps_col = "epsilon", g_col = "g"):
    df = pd.read_csv(csv_filename)

    eps_vals = np.array(sorted(df[eps_col].unique()))
    eps_mask = np.isclose(df[eps_col].astype(float).values, epsilon, atol = 1e-6)
    sub = df.loc[eps_mask, [eps_col, g_col]].copy()

    g_c = float(sub[g_col].min())
    return g_c
    

def append_higher_layers(
        g_values,
        phase_csv, base_params = {},
        threshold_list = [2.9, 3.8],
        eps_col = "epsilon", g_col = "g",
        layer_col = "layer", T = 50000,
        method = "LSODA", init_type = "gaussian",
        save_csv = None):
    # threshold_list provides threshold for height of 3, 4 and so on layers
    df = pd.read_csv(phase_csv).copy()
    
    df[layer_col] = 2
    epsilon_values = df[eps_col].values.astype(float)
    g_values = np.asarray(g_values, dtype = float)
    results = []
    print("Start simulation...")
    start = time.time()
    for eps in epsilon_values:
        # get critical growth rate

        g_c = get_g_c_from_epsilon(phase_csv, eps)
        start_idx = int(np.searchsorted(g_values, g_c, side = "right"))
        if start_idx >= len(g_values):
            continue

        next_thr_idx = 0
        for j in range(start_idx, len(g_values)):
            if next_thr_idx >= len(threshold_list):
                break
            
            g_val = g_values[j]
            if g_val >= 0.02:
                break
            # as g only until 10^-2 make educated decisions
            if eps >= 0.9:
                break
            if eps >= 0.25 and next_thr_idx == 1:
                break
            print(f"Start simulation with values epsilon={eps:.4g} and g={g_val:.4g}...")
            sim_start_time = time.time()
            params = {**base_params, "epsilon": eps, "g": g_val}
            t_event, h_center, _, _ = run_until_first_layer(
            params, T = T, method=method, init_type=init_type)
            sim_end_time = time.time()
            print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")

            if h_center >= threshold_list[next_thr_idx]:
                thr = threshold_list[next_thr_idx]
                results.append({
                    eps_col: eps,
                    g_col: g_val,
                    layer_col: int(thr)+1
                })
                print(f"Layer {next_thr_idx + 3} reached.")
                next_thr_idx += 1
            else:
                print(f"No transition for layer {next_thr_idx + 3} with h_max = {h_center:.2f} yet.")

    end = time.time()
    print(f"Simulation finished in {end - start:.2f}")
    out_df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok = True)
        out_df.to_csv(save_csv, index = False)
        print(f"Data saved to {save_csv}")

def append_higher_layers_bisect(
        g_upper,
        phase_csv,tol = 1e-5, base_params = {},
        threshold_list = [2.9, 3.8],
        eps_col = "epsilon", g_col = "g",
        layer_col = "layer", T = 50000,
        method = "LSODA", init_type = "gaussian",
        save_csv = None):
    # threshold_list provides threshold for height of 3, 4 and so on layers
    df = pd.read_csv(phase_csv).copy()
    
    df[layer_col] = 2
    epsilon_values = df[eps_col].values.astype(float)
    results = []
    print("Start simulation...")
    start = time.time()
    for eps in epsilon_values:
        # get critical growth rate

        next_thr_idx = 0
        g_c = get_g_c_from_epsilon(phase_csv, eps)
        
        print(f"\n[epsilon={eps:.5g}] g_c={g_c:.6g}, searching up to g_upper={g_upper:.6g}")

        while next_thr_idx < len(threshold_list):
            thr = threshold_list[next_thr_idx]
            # as g only until 10^-2 make educated decisions
            if eps >= 0.9:
                break
            if eps >= 0.25 and next_thr_idx == 1:
                break

            print(f"Start simulation with values epsilon={eps:.4g} and g={g_upper:.4g}...")
            sim_start_time = time.time()
            params = {**base_params, "epsilon": eps, "g": g_upper}
            t_event, h_center, _, _ = run_until_first_layer(
            params, T = T, method=method, init_type=init_type)
            sim_end_time = time.time()
            print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")

            if h_center < thr:
                print(f"Threshold for layer {next_thr_idx+3} (thr={thr}) not reached")

            # Start a bisection
            g_low, g_high = g_c, g_upper

            iterate = 0
            while(g_high - g_low) > tol:
                g_mid = 0.5 * (g_low + g_high)
                print(f"Start simulation with values epsilon={eps:.4g} and g={g_mid:.4g}...")
                sim_start_time = time.time()
                params = {**base_params, "epsilon": eps, "g": g_mid}
                t_event, h_mid, _, _ = run_until_first_layer(
                params, T = T, method=method, init_type=init_type)
                sim_end_time = time.time()
                print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")

                if h_mid >= thr:
                    g_high = g_mid
                else:
                    g_low = g_mid
                iterate += 1
                print(f"Interval width: {g_high - g_low:.2e}.")

            g_found = g_high
            layer_num = int(thr) + 1
            results.append({
                eps_col: eps,
                g_col: g_found,
                layer_col: layer_num
            })

            print(f"Layer {next_thr_idx + 3} reached at g = {g_found}.")
            next_thr_idx += 1


    end = time.time()
    print(f"Simulation finished in {end - start:.2f}")
    out_df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok = True)
        out_df.to_csv(save_csv, index = False)
        print(f"Data saved to {save_csv}")

def plot_layer_phase_diagram(
        data, 
        x_col = "epsilon", y_col = "g", layer_col = 'layer',
        cmap = 'viridis'):
    df = pd.read_csv(data)

    layers = sorted(df[layer_col].unique())

    fig, ax = plt.subplots()

    cmap_obj = get_cmap(cmap)
    nL = len(layers)

    layers_to_colors = {
        L: cmap_obj(i / (nL - 1)) for i, L in enumerate(layers)
    }

    curves = {}
    for L in layers:
        sub = df[df[layer_col] == L].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(by=x_col)
        
        x = sub[x_col].values
        y = sub[y_col].values
        curves[L] = (x, y)

    for L in layers:
        x, y = curves[L]
        ax.plot(x, y, label = f"Layer {L}")

    # Fill between adjacent layer curves (L and next_L)
    def _fill_between_curves(x1, y1, x2, y2, color):
        """
        Interpolate both curves on the *overlapping* x-range,
        then fill_between. Linear interpolation (monotone-safe).
        """
        # Determine overlap
        xmin = max(x1.min(), x2.min())
        xmax = min(x1.max(), x2.max())
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return  # no overlap

        # Build interpolation grids within overlap
        xgrid = np.linspace(xmin, xmax, 101)

        # Interpolants (assume x arrays are sorted)
        y1i = np.interp(xgrid, x1, y1)
        y2i = np.interp(xgrid, x2, y2)

        # Choose which is "upper" and "lower" for visual consistency:
        # typically higher layer threshold requires larger g (upper curve).
        y_upper = np.maximum(y1i, y2i)
        y_lower = np.minimum(y1i, y2i)

        ax.fill_between(xgrid, y_lower, y_upper, color=color, linewidth=0)


    xmin = max(x1.min(), x2.min())
    xmax = min(x1.max(), x2.max())
    x0, y0 = curves[layers[0]]
    xgrid = np.linspace(xmin, xmax, 101)
    y1i = np.interp(xgrid, x1, y1)
    y2i = np.interp(xgrid, x2, y2)
    y_upper = np.maximum(y1i, y2i)
    y_lower = np.minimum(y1i, y2i)
    ax.fill_between(xgrid, y_lower, y_upper, color=layers_to_colors[0], linewidth=0)

    for i in range(len(layers) - 1):
        L_low = layers[i]
        L_high = layers[i + 1]
        if L_low in curves and L_high in curves:
            x1, y1 = curves[L_low]
            x2, y2 = curves[L_high]
            _fill_between_curves(x1, y1, x2, y2, color = layers_to_colors[L_high])
    
    ax.set_xlabel("Energy scale $\epsilon$")
    ax.set_ylabel("Growth rate $g$")
    ax.legend()
    plt.show()
        

if __name__ == "__main__":
    csv_filename = "Results/data/phase_transition_g_eps.csv"
    g_values = np.logspace(np.log10(0.0005), -1, 201)
    save_csv = "Results/data/phase_diagram.csv"
    #append_higher_layers(g_values= g_values, phase_csv=csv_filename, save_csv=save_csv)
    append_higher_layers_bisect(g_upper = 0.02, phase_csv=csv_filename,
                                save_csv=save_csv)
