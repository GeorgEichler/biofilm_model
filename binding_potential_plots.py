import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 0.2 / (x**2)


def g(x):
    a = 0.25
    # h(x) = 1.5*cos(2πx + 0.3π)*e^{-x/1} + 10*e^{-x/0.01}
    return 1.5 * np.cos(2.0 * np.pi * (x-a) + 0.3 * np.pi) * np.exp(-(x-a)) + 10.0 * np.exp(-(x-a) / 0.01)

def h(x):
    # g(x) = 0.5 * ( -1/(2 x^2) + 0.3^3/(5 x^5) )
    return 0.5 * ( -1.0 / (2.0 * x**2) + (0.3**3) / (5.0 * x**5) )

def k(x):
    b = -0.5
    #return 3*(1 / (x-b)**5 - 1/(x-b)**2 + np.exp(-(x-b)/3)*0.5)
    return (1 / (x-b)**5 - 1.5/(x-b)**2 + np.exp(-(x-b)/3)**0.5)


def main():
    plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.dpi": 100 #change resolution, standard is 100
        })
    # Domain (avoid x=0 to prevent division-by-zero)
    x_min, x_max = 0.1, 5
    x = np.linspace(x_min, x_max, 10001)

    y_f = f(x)
    y_k = k(x)
    y_h = h(x)
    y_lo = -2
    y_hi = 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    # Plot f
    axes[0].plot(x, y_f)
    #axes[0].grid(True, alpha=0.3)

    # Plot g
    axes[1].plot(x, y_h)
    #axes[1].grid(True, alpha=0.3)

    # Plot h
    axes[2].plot(x, y_k)
    #axes[2].grid(True, alpha=0.3)

    # Shared axes labels & limits
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_lo, y_hi)
        ax.axhline(0, color="black", linewidth=1, linestyle="-")
    fig.supxlabel("h")
    fig.supylabel("f(h)")

    fig.tight_layout()
    plot_filename = "Results/plots/binding_potentials.png"
    plt.savefig(plot_filename, dpi = 300, bbox_inches = 'tight')
    print(f"file saved to {plot_filename}")
    plt.show()

if __name__ == "__main__":
    main()
