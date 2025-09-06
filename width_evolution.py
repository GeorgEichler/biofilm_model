import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import time
import seaborn as sns
import itertools
import os
import csv

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model
from helper_functions import find_first_k_minima

# models
def _linear(x, a, b):
    return a*x + b

def _power(x, C, p, d):
    return C * np.power(x, p) + d

def calculate_width_at_center(h, x, h_low):
    """
    Calculates the width of interval containing the center where
    the film height surpasses the first layer
    """

    H = np.asarray(h)
    x = np.asarray(x)

    if H.ndim != 2:
        raise ValueError("H must be a 2D array with shape (N_space, times)")

    N, M = H.shape
    
    center = N // 2
    mask = H >= h_low
    widths = np.zeros(M, dtype = float)
    left_idx = np.full(M, center, dtype = int)
    right_idx = np.full(M, center, dtype = int)
    x_left = np.full(M, x[center], dtype = float)
    x_right = np.full(M, x[center], dtype = float)

    for j in range(M):
        if not mask[center, j]:
            # center below threshold
            continue

        # ----- scan left (inclusive) -----
        i = center
        while i - 1 >= 0 and mask[i - 1, j]:
            i -= 1
        left_idx[j] = i

        # ----- scan right (inclusive) -----
        i = center
        while i + 1 < N and mask[i + 1, j]:
            i += 1
        right_idx[j] = i

        # Make interpolation between grid ponts
        xl = x[left_idx[j]]
        xr = x[right_idx[j]]

        
        col = H[:, j]

        # Left crossing between (left_idx-1) -- (left_idx), if possible
        if left_idx[j] > 0:
            h0, h1 = col[left_idx[j] - 1], col[left_idx[j]]
            x0, x1 = x[left_idx[j] - 1], x[left_idx[j]]
            denom = (h1 - h0)
            # We expect h0 < h_low <= h1 (typical), but guard anyway:
            if denom != 0:
                t = (h_low - h0) / denom  # fraction from x0->x1
                # clamp for numerical safety
                t = max(0.0, min(1.0, t))
                xl = x0 + t * (x1 - x0)

        # Right crossing between (right_idx) -- (right_idx+1), if possible
        if right_idx[j] < N - 1:
            h0, h1 = col[right_idx[j]], col[right_idx[j] + 1]
            x0, x1 = x[right_idx[j]], x[right_idx[j] + 1]
            denom = (h1 - h0)
            if denom != 0:
                t = (h_low - h0) / denom  # fraction from x0->x1
                t = max(0.0, min(1.0, t))
                xr = x0 + t * (x1 - x0)

        x_left[j] = xl
        x_right[j] = xr

        widths[j] = float(x[right_idx[j]] - x[left_idx[j]])

    widths = np.minimum(widths, 100)

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

def plot_widths(sweep_params, base_params = {}, T = 1000,
                method = 'LSODA', init_type = 'gaussian',
                plot_filename = None, csv_filename = None):
    """
    Simulation for calculating the width of the first layer of the biofilm
    """

    _, ax = plt.subplots()

    param_keys = list(sweep_params.keys())
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

        times, widths = calculate_width_evolution(
            full_params, T=T, method=method, init_type=init_type
        )
        label = ", ".join([f"{k}={v:.3g}" for k, v in current_sweep_params.items()])
        ax.plot(times, widths, label = label)

        for t, w in zip(times, widths):
            row = {**current_sweep_params, "t": t, "width": w}
            results.append(row)
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

    if csv_filename:
        output_dir = os.path.dirname(csv_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(results)
        df = df[param_keys + ["t", "width"]]
        df.to_csv(csv_filename, index = False)
        print(f"Data saved to {csv_filename}")

    plt.show()

def replot_width(csv_filename, csv_filename2 = None, plot_filename = None, min_width = None, t_min = None,
                 t_max = None, fit = False, x_col="t", y_col="width", series_params=None,
                 xlog=False, ylog = False, scaling = False, extend_to_t = 20000, y_max = 100):
    """
    Replot saved results. If series_params is None, uses all columns except x,y.
    """
    df = pd.read_csv(csv_filename)

    # Apply scaling rule
    if scaling:
        g_vals = df["g"].values
        t_vals = df[x_col].values

        df["t_scaled"] = g_vals * t_vals

        x_col = "t_scaled"

    if xlog:
        df = df[df[x_col] >= 1]

    if ylog:
        df = df[df[y_col] >= 1]

    if series_params is None:
        exclude = {x_col, y_col, "t"}
        series_params = [c for c in df.columns if c not in exclude]

    # group by parameter combinations
    grouped = df.groupby(series_params, dropna=False)
    palette = sns.color_palette("colorblind", 5) # make a colorblind save scale
    i = 0
    fig, ax = plt.subplots()
    for keys, sub in grouped:
        # build label
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = ", ".join(f"{k}={v:g}" for k, v in zip(series_params, keys))
        sub_sorted = sub.sort_values(x_col)
        x_plot = sub_sorted[x_col].values
        y_plot = sub_sorted[y_col].values
        ax.plot(x_plot, y_plot, label=label, color = palette[i])
        i += 1

        if fit:
            fit_sub = sub_sorted.copy()
            fit_sub = sub.drop_duplicates(subset=[y_col], keep="first")
            #sub = sub.sort_values(x_col)

            # trim width (optional)
            if min_width:
                fit_sub = fit_sub[fit_sub[y_col] > min_width]

            if t_min is not None:
                fit_sub = fit_sub[sub[x_col] >= t_min]
            if t_max is not None:
                fit_sub = fit_sub[fit_sub[x_col] <= t_max]
            
            x = fit_sub[x_col].values
            y = fit_sub[y_col].values
            a0 = (y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 0.0
            b0 = y.mean() - a0 * x.mean()
            try:
                (a_hat, b_hat), _ = curve_fit(_linear, x, y, p0=(a0, b0), maxfev=10000)
                xfit = np.linspace(x.min(), x.max(), 200)
                ax.plot(xfit, _linear(xfit, a_hat, b_hat), linestyle="--",
                        label=f"Linear fit ({label})")
                print(f"Linear fit {label}: y = {a_hat:.6g} * t + {b_hat:.6g}")
            except Exception as e:
                print(f"[WARN] Linear fit failed for {label}: {e}")
            
            pmask = (x > 0) & (y > 0)
            
            xp, yp = x[pmask], y[pmask]
            # rough initial guesses: C0 ≈ y0 / x0, p0 ≈ 1
            C0 = max(yp[0] / max(xp[0], 1e-12), 1e-12)
            p0 = 1.0
            d0 = 0
            try:
                (C_hat, p_hat, d_hat), _ = curve_fit(
                    _power, xp, yp, p0=(C0, p0, d0),
                    bounds=([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf])  # keep C ≥ 0
                )
                xfit = np.linspace(xp.min(), xp.max(), 400)
                ax.plot(xfit, _power(xfit, C_hat, p_hat, d_hat), linestyle=":",
                        label=f"Power fit ({label})")
                print(f"Power-law fit {label}: y = {C_hat:.6g} * t^{p_hat:.6g}")
            except Exception as e:
                print(f"[WARN] Power-law fit failed for {label}: {e}")

    if csv_filename2:
        df2 = pd.read_csv(csv_filename2)
        if series_params is None:
            exclude = {x_col, y_col, "t"}
            series_params = [c for c in df.columns if c not in exclude]

        # group by parameter combinations
        grouped2 = df2.groupby(series_params, dropna=False)
        for keys, sub in grouped2:
            # build label
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = ", ".join(f"{k}={v:g}" for k, v in zip(series_params, keys))
            sub_sorted = sub.sort_values(x_col)
            x_plot = sub_sorted[x_col].values
            y_plot = sub_sorted[y_col].values
            x_plot = np.r_[x_plot, extend_to_t]
            y_plot = np.r_[y_plot, y_max]
            ax.plot(x_plot, y_plot, label=label, color = palette[i])
            i +=1
        
    ax.set_xlim(left = 0)
    ax.set_ylim(bottom = 0)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Width (w)")
    if scaling:
        ax.set_xlim(0, 20)
        ax.set_xlabel("$t*g$")
        ax.set_ylabel("Width of biofilm (w)")
    if xlog:
        ax.set_xscale('log')
        ax.set_xlim(left = 1)
    if ylog:
        ax.set_yscale('log')
        ax.set_ylim(bottom = 1)
    ax.legend()

    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok = True)
        plt.savefig(plot_filename, dpi = 300, bbox_inches = 'tight')
        print(f"Saved figure to: {plot_filename}")

    plt.show()

if __name__ == "__main__":
    choice = input("Width simulation (a) or replot from data (b): ")
    
    if choice == "a":
        # Use odd number of grid points to get the middle grid
        base_params = {
            'L': 200,
            'N': 2049
        }
        sweep_params = {
            'g': [1e-4, 1e-3]
        }

        plot_filename = "Results/plots/width_evolution_g_L200_monolayer.png"
        csv_filename = "Results/data/width_evolution_g_L200_monolayer.csv"

        plot_widths(
            sweep_params=sweep_params,
            T = 20000,
            base_params=base_params,
            plot_filename=plot_filename,
            csv_filename=csv_filename
        )

    elif choice == "b":
        plot_filename = "Results/plots/width_evolution_both_regimes.png"
        csv_filename = "Results/data/width_evolution_g_L200_monolayer.csv"
        csv_filename2 = "Results/data/width_evolution_g.csv"

        replot_width(csv_filename=csv_filename, csv_filename2=csv_filename2, plot_filename=plot_filename,
                     xlog = False, scaling=False, min_width=5, fit = False)