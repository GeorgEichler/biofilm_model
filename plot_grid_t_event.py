from layer_formation_analysis import run_until_first_layer
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from matplotlib.ticker import ScalarFormatter

def plot_event_profiles_grid(
        g_values, epsilon_values,
        base_params = {}, T = 50000,
        method = 'LSODA', init_type = 'gaussian',
        plot_filename = None,
        csv_filename = None):
    g_values = list(g_values)
    epsilon_values = list(epsilon_values)

    ncols = len(epsilon_values)
    nrows = len(g_values)
    total_sims = ncols * nrows
    print(f"Starting parameter sweep with {total_sims} simulations...")
    start_time = time.time()

    fig, axes = plt.subplots(nrows, ncols, figsize = (10, 8), sharex=True, sharey=True)
    results = []

    for i, g in enumerate(g_values):
        for j, eps in enumerate(epsilon_values):
            #ax = axes[len(g_values) - 1 - i, j]
            # set this if axis is one dimensional
            ax = axes[j]
            params = {**base_params, 'g': g, 'epsilon': eps}

            param_str = ', '.join([f"{k}={v:.4g}" for k, v in params.items()])
            print(f'[{i+j+1}/{total_sims}] Running with {param_str}...')
        
            sim_start_time = time.time()

            t_event, _, h_event, x = run_until_first_layer(
                params=params, T = T, method=method, init_type=init_type
            )
            sim_end_time = time.time()
            print(f" -> Time for this step: {sim_end_time - sim_start_time:.2f}s.")


            row = {
                'g': g,
                'epsilon': eps,
                't_event': t_event,
                **{f"h_{k}": val for k, val in enumerate(h_event)}
            }
            results.append(row)
            ax.plot(x, h_event)
            ax.set_xlim(0, 100)
            ax.set_ylim(bottom = 0)

            if i == 0:
                ax.set_xlabel("x")
            if j == 0:
                ax.set_ylabel("$h(x, t_{event})$")

    end_time = time.time()
    print(f"\nFull simulation time: {end_time - start_time:.2f}s.")

    if csv_filename:
        df = pd.DataFrame(results)
        df.to_csv(csv_filename, index = False)
        print(f"Data saved to: {csv_filename}")

    if plot_filename:
        outdir = os.path.dirname(plot_filename)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {plot_filename}")

    plt.show()

def replot_event_profiles_from_csv(csv_filename, x, g_values, epsilon_values,
                                   plot_filename = None):
    df = pd.read_csv(csv_filename)

    ncols, nrows = len(epsilon_values), len(g_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 10.5), sharex=True, sharey=True)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_scientific(True)
    sci_formatter.set_powerlimits((0, 0))  # force sci notation

    for i, g in enumerate(g_values):
        for j, eps in enumerate(epsilon_values):
            ax = axes[nrows - 1 - i, j]
            # use if axis only one dimensional
            #ax = axes[j]
            row = df[(df['g']==g) & (df['epsilon']==eps)]
            if row.empty or pd.isna(row['t_event'].values[0]):
                ax.text(0.5, 0.5, 'No event', ha='center', va='center', transform=ax.transAxes)
                continue

            # Extract profile back from columns h_0, h_1, ...
            h_cols = [c for c in row.columns if c.startswith("h_")]
            h_event = row[h_cols].values[0]
            ax.plot(x, h_event)
            #ax.set_title(f"g={g}, ε={eps}, $t_{{event}}$={row['t_event'].values[0]:.3g}")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 5.5)

            g_str = sci_formatter.format_data(g)

            if i == 0:
                ax.set_xlabel(fr"x, $\epsilon$ = {eps}")
            if j == 0:
                if g == 10:
                    ax.set_ylabel(fr"$h(x, t_{{complete}})$" +"\n" + fr"$g = {g}$")
                else:
                    ax.set_ylabel(fr"$h(x, t_{{complete}})$" +"\n" + fr"$g = {g_str}$")

    if plot_filename:
        outdir = os.path.dirname(plot_filename)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {plot_filename}")

    plt.show()

if __name__ == "__main__":
    choice = input("Profile simulation (a) or plotting (b): ")
    # make this more general
    g_values = [1e-3, 1e-2, 1e-1, 1, 10]
    #g_values = [1e-2]
    epsilon_values = [0.5, 1, 1.5, 2]

    if choice == "a":

        csv_filename = "Results/data/event_profiles_g1e-2.csv"
        plot_event_profiles_grid(g_values, epsilon_values, csv_filename=csv_filename)
    
    elif choice == "b":
        # get grid for plotting
        _, _, _, x = run_until_first_layer({'epsilon': 0, 'g': 10})
        csv_filename = "Results/data/event_profiles.csv"
        #csv_filename = "Results/data/event_profiles_g1e-2.csv"
        plot_filename = "Results/plots/event_profiles.png"
        replot_event_profiles_from_csv(csv_filename, x, g_values, epsilon_values, plot_filename=plot_filename)