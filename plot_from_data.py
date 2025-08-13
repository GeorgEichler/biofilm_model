import pandas as pd
import matplotlib.pyplot as plt


def plot_event_from_csv(csv_filename):
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

    # Format height plot
    ax_h.set_xscale('log')
    ax_h.set_xlabel('g')
    ax_h.set_ylabel('h(t_event, L/2)')
    ax_h.set_title('Height at middle of profile')
    ax_h.legend()

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
    
    plot_event_from_csv("Results/data/first_layer_g_eps.csv")
    