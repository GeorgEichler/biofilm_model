import numpy as np
import matplotlib.pyplot as plt

class FigureHandler:
    """
    Handling of plots for the thin-film equation model
    """
    def __init__(self, model):
        self.model = model
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })

    def plot_binding_energy(self, g, h_min = 0, h_max = 10, nh = 1001):
        h_array = np.linspace(h_min, h_max, nh)
        plt.figure()
        plt.plot(h_array, g(h_array))
        plt.xlabel('h')
        plt.ylabel('g(h)')
        plt.title('Binding potential')

    def plot_profiles(self, H, times, pot_minima = None):
        x = self.model.x # get grid of model
        plt.figure()
        for h, t in zip(H.T, times):
            plt.plot(x, h, label=f't={t:.2f}')
        if pot_minima is not None:
            for y in pot_minima:
                plt.hlines(y, xmin=x[0], xmax=x[-1], linestyles='dashed')
        plt.xlabel('x')
        plt.ylabel('h(x,t)')
        plt.legend(loc = 'right')
        plt.grid(True)

    def plot_free_energy(self, H, times):
        # Convert to numpy array to use slice operations
        energy_values = np.array([self.model.free_energy(H[:,i]) for i in range(len(times))])
        surface_values = energy_values[:, 0]
        potential_values = energy_values[:, 1]
        plt.figure()
        plt.plot(times, surface_values, '--', label = 'Surface energy')
        plt.plot(times, potential_values, '--', label = "Potential energy")
        plt.plot(times, surface_values + potential_values, '-o', label = "Total energy")
        plt.xlabel('t')
        plt.ylabel('E')
        plt.title('Free energy evolution')
        plt.grid(True)
        plt.legend()