import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_first_layer_event_from_csv(csv_filename, filename_event_time = None, filename_height = None):
    """
    Load simulation data from CSV and generate plots of first reached layer
    """

    # Load CSV into dataframe
    df = pd.read_csv(csv_filename)
    if df.empty:
        print("CSV file is empty. Nothing to plot.")
        return
    
    print(f"Loaded {len(df)} rows from {csv_filename}.")

    # Sort values for plotting
    df = df.sort_values(['epsilon', 'g'])

    fig_t, ax_t = plt.subplots()
    fig_h, ax_h = plt.subplots()

    for eps, group in df.groupby('epsilon'):
        group = group.sort_values('g')
        g = group['g'].values
        t_event = group['t_event'].values
        h_center = group['h_center'].values

        label = rf'$\epsilon = {eps}$'
        ax_t.plot(g, t_event, marker = 'o', linestyle = '-', label = label)
        ax_h.plot(g, h_center, marker = 'o', linestyle = '-', label = label)

    ax_t.set_xscale('log')
    ax_t.set_xlabel('g')
    ax_t.set_ylabel('t_event')
    ax_t.set_title('Time to first layer')
    ax_t.legend()

    if filename_event_time:
        fig_t.savefig(filename_event_time, dpi = 300, bbox_inches = 'tight')

    # Format height plot
    ax_h.set_xscale('log')
    ax_h.set_xlabel('Growth parameter (g)')
    ax_h.set_ylabel('h(t_event, L/2)')
    #ax_h.set_title('Height at middle of profile')
    ax_h.legend()

    if filename_height:
        fig_h.savefig(filename_height, dpi = 300, bbox_inches = 'tight')

    plt.show()    

def plot_critical_time_event_from_csv(csv_filename, xaxis_param='g', series_params=None, plot_filename=None):
    """
    Recreate the plot of critical time vs growth parameter from saved CSV.
    
    Args:
        csv_filename (str): Path to CSV file (must contain columns including `t_event`).
        xaxis_param (str): Parameter to use on the x-axis (default: 'g').
        series_params (list[str] | None): Parameters to separate into different series (default: infer from CSV).
        plot_filename (str | None): If given, save plot to this file.
    """
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"No such file: {csv_filename}")

    # Load CSV
    df = pd.read_csv(csv_filename)
    print(f"Loaded {len(df)} rows from {csv_filename}.")
    print("Columns:", df.columns.tolist())

    # If not given, infer series parameters from columns (exclude x and t_event)
    if series_params is None:
        series_params = [col for col in df.columns if col not in [xaxis_param, 't_event']]

    _, ax = plt.subplots()

    if series_params:
        grouped = df.groupby(series_params)
        for series_key, group in grouped:
            group = group.sort_values(by=xaxis_param)
            x_vals = group[xaxis_param].values
            t_vals = group['t_event'].values
            if not isinstance(series_key, tuple):
                series_key = (series_key,)
            label = ", ".join([f"{p}={v}" for p, v in zip(series_params, series_key)])
            ax.plot(x_vals, t_vals, marker='o', linestyle='-', label=label)
    else:
        group = df.sort_values(by=xaxis_param)
        ax.plot(group[xaxis_param], group['t_event'], marker='o', linestyle='-', label=None)

    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel("Growth parameter (g)")
    ax.set_ylabel('Critical time ($t_c$)')
    ax.legend()

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_filename}")

    plt.show()

def plot_width_from_csv(csv_filename,
                        save_filename=None,
                        logx=False,
                        logy=False,
                        xlim=None,
                        ylim=None):
    """
    Load (time, width, <param columns...>) data from CSV and plot width vs time
    for each parameter constellation.

    Parameters
    ----------
    csv_filename : str
        Path to the CSV produced by your sweep (columns: time, width, and params).
    save_filename : str or None
        If provided, save the figure here.
    title : str
        Plot title.
    logx : bool
        Use log scale on x-axis (time) if True.
    xlim, ylim : tuple or None
        Axis limits, e.g. (0, 2500).
    """
    df = pd.read_csv(csv_filename)
    if df.empty:
        print("CSV is empty – nothing to plot.")
        return

    # Make sure standard columns exist
    required = {'time', 'width'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain {required}, found {set(df.columns)}")

    # Identify sweep parameter columns automatically
    param_cols = [c for c in df.columns if c not in ('time', 'width')]
    if not param_cols:
        print("No sweep parameters detected; plotting a single curve.")
        param_cols = []  # just one group

    # Sort for clean lines
    df = df.sort_values(['time'] + param_cols if param_cols else ['time'])

    fig, ax = plt.subplots()

    # Group by all parameter columns (or single group if none)
    if param_cols:
        grouped = df.groupby(param_cols, dropna=False)
    else:
        # single pseudo-group
        grouped = [((), df)]

    for key, group in grouped:
        # key is either a tuple of param values aligned with param_cols or ()
        label = ", ".join(f"{k}={v:g}" for k, v in zip(param_cols, key)) if param_cols else "run"
        # Ensure time-sorted within group
        group = group.sort_values('time')
        ax.plot(group['time'].values, group['width'].values, label=label)

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Width (w)')
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if xlim is not None:
        ax.set_xlim(left=xlim[0],right=xlim[1])
    if ylim is not None:
        ax.set_ylim(*ylim)
    if param_cols:
        ax.legend()

    fig.tight_layout()

    if save_filename:
        out_dir = os.path.dirname(save_filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_filename}")

    plt.show()

def plot_widths_from_csv(csv_filename, plot_filename=None):
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"No such file: {csv_filename}")

    # Load CSV into dataframe
    df = pd.read_csv(csv_filename)

    # Quick sanity check
    print(f"Loaded {len(df)} rows from {csv_filename}.")
    print("Columns:", df.columns.tolist())

    # Extract columns
    times = df['t'].values
    widths1 = df['First layer'].values
    widths2 = df['Second layer'].values
    widths3 = df['Third layer'].values

    # Make the plot
    fig, ax = plt.subplots()
    ax.plot(times, widths1, label='1st layer')
    ax.plot(times, widths2, label='2nd layer')
    ax.plot(times, widths3, label='3rd layer')

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Width (w)')
    ax.legend()
    ax.grid(True)

    if plot_filename:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_filename}")

    plt.show()

def plot_phase_transition(csv_filename, plot_filename = None):
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"No such file: {csv_filename}")
    
    df = pd.read_csv(csv_filename)
    print(f"Loaded {len(df)} rows from {csv_filename}.")
    print("Columns:", df.columns.tolist())

    # Extract columns
    g_values = df['g'].values
    epsilon_values = df['epsilon'].values

    plt.figure()
    plt.plot(epsilon_values, g_values, marker = 'o', linestyle = '-')
    plt.xlabel("Strength parameter ($\epsilon$)")
    plt.ylabel("Growth parameter (g)")
    #plt.yscale('log')

    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok = True)
        plt.savefig(plot_filename, dpi = 300, bbox_inches = 'tight')
        print(f"Saved figure to: {plot_filename}")
    plt.show()

if __name__ == "__main__":
    plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })
    
    csv_path = "Results/data/critical_time_eps.csv"
    plot_critical_time_event_from_csv(
        csv_filename=csv_path
    )
    exit()

    csv_path = "Results/data/phase_transition_g_eps.csv"
    plot_filename = "Results/plots/phase_transition_eps_g_no_log.png"
    plot_phase_transition(csv_filename = csv_path, plot_filename=plot_filename)

    csv_path = "Results/data/width_evolution_g002.csv"

    plot_width_from_csv(
        csv_filename=csv_path,
        save_filename="Results/plots/width_evolution_g002.png",
        logx=False,
        logy=False,
    )

    filename_event_time = "Results/plots/event_time_long_range_g_eps.png"
    filename_height = "Results/plots/height_long_range_g_eps.png"
    
    plot_first_layer_event_from_csv("Results/data/first_layer_long_range_eps.csv",
                        filename_event_time=filename_event_time,
                        filename_height=filename_height)
    