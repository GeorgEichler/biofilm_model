import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import os
import itertools

from OneD_FDM_simple_model import FDM_OneD_Thin_Film_Model


def get_max_height_timeseries(params, T, init_type = 'gaussian', method = 'LSODA'):
    """
    Runs a single thin film evolution and returns the times series of the height
    at the center of the domain x = L/2

    Args:
        params (dict): A dictionary of parameters for the model.
        T (float): The total simulation time.
        t_eval (np.ndarray): Array of time points to evaluate the solution at.
        init_type (str): The type of initial condition.
        method (str): The solver method for solve_ivp.
    Returns:
        tuple: (times, h_center_series) where `times` is the array of time points
               and `h_center_series` is the corresponding height at x=L/2.
    """

    print("-" * 50)
    param_str = ', '.join([f"{k}={v:.4g}" for k, v in params.items()])
    print(f"Running simulation with: {param_str}")
    
    # 1. Initialize the model with the given parameters
    model = FDM_OneD_Thin_Film_Model(**params)

    h_init = model.setup_initial_conditions(init_type)

    times, H = model.solve(h_init, T = T, method = method)

    # get maximum value of series
    h_max_series = np.minimum(np.max(H, axis=0), 5.0)

    return times, h_max_series


def plot_max_height_evolution(sweep_params, base_params = {}, T = 1000,
                                 plot_filename = None, csv_filename = None):
    """
    Perform a parameter sweep and plot the time evolution at the center height
    """

    _, ax = plt.subplots()

    param_keys = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    param_combinations = list(itertools.product(*param_values))
    total_sims = len(param_combinations)

    print(f"Starting sweep with {total_sims} simulations...")
    start_time = time.time()

    results = []

    for i, combo in enumerate(param_combinations):
        current_sweep_params = dict(zip(param_keys, combo))
        full_params = {**base_params, **current_sweep_params}


        times, h_max_series = get_max_height_timeseries(
            full_params, T)


        label = ", ".join([f"{k}={v:.3g}" for k, v in current_sweep_params.items()])
        ax.plot(times, h_max_series, label=label)

        for t_val, h_val in zip(times, h_max_series):
            row = {**current_sweep_params, "t": t_val, "h_max": h_val}
            results.append(row)

    end_time = time.time()
    print(f"\nTotal sweep finished in {end_time - start_time:.2f}s.")


    ax.set_xlabel('Time (t)')
    ax.set_ylabel('$h_{max}(t)$')
    #ax.set_title('Evolution of Center Height for Different Parameters')
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
            os.makedirs(output_dir, exist_ok = True)
        df = pd.DataFrame(results)
        df = df[param_keys + ["t", "h_max"]]
        df.to_csv(csv_filename, index = False)
        print(f"Data saved to {csv_filename}")

    plt.show()

def replot_max_height(csv_filename, plot_filename = None, 
                      x_col="t", y_col="h_max", series_params=None, loglog=False, scaling = False):
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

    if loglog:
        df = df[df[x_col] >= 1]

    if series_params is None:
        exclude = {x_col, y_col, "t"}
        series_params = [c for c in df.columns if c not in exclude]

    # group by parameter combinations
    grouped = df.groupby(series_params, dropna=False)

    fig, ax = plt.subplots()
    palette = sns.color_palette("colorblind", 5)
    i = 0
    for keys, sub in grouped:
        # build label
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = ", ".join(f"{k}={v:g}" for k, v in zip(series_params, keys))
        sub_sorted = sub.sort_values(x_col)
        linestyle = "-"
        if keys[0] == 1e-4:
            linestyle = (0, (5, 5)) # dashed linestyle
        if keys[0] == 1e-3:
            linestyle = (5, (5, 5)) # dashed linestyle with shifted phase
        ax.plot(sub_sorted[x_col].values, sub_sorted[y_col].values, label=label,
                linestyle = linestyle, color = palette[i])
        i += 1

    ax.set_xlim(left = 0)
    ax.set_ylim()
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("$h_{max}(t)$")
    if scaling:
        ax.set_xlabel("$tg$")
        ax.set_xlim(0, 25)
    if loglog:
        ax.set_xscale('log')
        ax.set_xlim(left = 1)
    ax.legend()

    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok = True)
        plt.savefig(plot_filename, dpi = 300, bbox_inches = 'tight')
        print(f"Saved figure to: {plot_filename}")

    plt.show()

def replot_max_height_seaborn(csv_filename, plot_filename = None, 
                      x_col="t", y_col="h_max", series_params=None, loglog=False, scaling = False):
    """
    Replot saved results. If series_params is None, uses all columns except x,y.
    """
    df = pd.read_csv(csv_filename)
    g_array = df["g"].values.unique()
    # Apply scaling rule
    if scaling:
        g_vals = df["g"].values
        t_vals = df[x_col].values

        df["t_scaled"] = g_vals * t_vals

        x_col = "t_scaled"

    if loglog:
        df = df[df[x_col] >= 1]

    if series_params is None:
        exclude = {x_col, y_col, "t"}
        series_params = [c for c in df.columns if c not in exclude]

    # group by parameter combinations
    grouped = df.groupby(series_params, dropna=False)

    sns.set_style("white")

    fig, ax = plt.subplots()
    norm = plt.Normalize(df["g"].min(), df["g"].max())
    cmap = sns.color_palette(palette = "crest", as_cmap=True)
    for keys, sub in grouped:
        # build label
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = ", ".join(f"{k}={v:g}" for k, v in zip(series_params, keys))
        sub_sorted = sub.sort_values(x_col)
        color = cmap(norm(keys))
        linestyle = "solid"
        if keys == 1e-4:
            linestyle = "-"
        if keys == 1e-3:
            linestyle = '--'
        ax.plot(sub_sorted[x_col].values, sub_sorted[y_col].values, color = color,
                linestyle = linestyle)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax = ax)
    cbar.set_ticks(g_array)
    cbar.set_ticklabels([f"{g}" for g in g_array])
    cbar.set_label("g")
    ax.set_ylim()
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("$h_{max}(t)$")
    if scaling:
        ax.set_xlabel("$t*g$")
        ax.set_xlim(0, 25)
    if loglog:
        ax.set_xscale('log')
        ax.set_xlim(left = 1)
    ax.legend()

    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok = True)
        plt.savefig(plot_filename, dpi = 300, bbox_inches = 'tight')
        print(f"Saved figure to: {plot_filename}")

    plt.show()

if __name__ == '__main__':

    choice = input("Do max height evolution (a) or replot from data (b): ")

    if choice == "a":
        base_params = {
            'L': 200,
            'N': 2048
        }
        sweep_params = {
            'g': [1e-4, 1e-3, 1e-2, 1e-1, 1]
        }

        plot_filename = "Results/plots/max_height_evolution_g_both_regimes.png"
        csv_filename = "Results/data/max_height_evolution_g_both_regimes.csv"

        plot_max_height_evolution(
            T = 2500,
            base_params=base_params,
            sweep_params=sweep_params,
            plot_filename=plot_filename,
            csv_filename=csv_filename
        )

    elif choice == "b":
        #csv_filename = "Results/data/max_height_evolution_g_both_regimes.csv"
        #plot_filename = "Results/plots/max_height_evolution_g_both_regimes.png"
        csv_filename = "Results/data/max_height_evolution_g_t2500.csv"
        plot_filename = "Results/plots/max_height_evolution_g_t2500_log.png"

        replot_max_height(csv_filename = csv_filename, plot_filename=plot_filename, loglog=True, scaling=False)
        