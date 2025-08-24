import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from OneD_thin_film_model import OneD_Base_Model
class FigureHandler:
    """
    Handling of plots for the thin-film equation model
    """
    def __init__(self, model:OneD_Base_Model, output_dir: str = "Results/plots"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok = True)

        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })

    def save_figure(self, filename: str):
        if not filename.endswith(".png"):
            filename += ".png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi = 300, bbox_inches = 'tight')
        print(f"Saved figure: {path}")

    def plot_binding_energy(self, f, h_min = 0, h_max = 5, nh = 1001, filename = None):
        h_array = np.linspace(h_min, h_max, nh)
        plt.figure()
        plt.plot(h_array, f(h_array))
        plt.xlabel('h')
        plt.ylabel('f(h)')
        plt.xlim(0, 5)
        plt.ylim(-0.5, 2)
        #plt.title('Binding potential')
        if filename:
            self.save_figure(filename)

    def plot_growth_function(self, g, h_min = 0, h_max = 5.1, nh = 1001, filename = None):
        h_array = np.linspace(h_min, h_max, nh)
        plt.figure()
        plt.plot(h_array, g(h_array))
        plt.xlabel('h')
        plt.ylabel('G(h)')
        plt.xlim(0, h_max)
        #plt.title("Growth function")
        plt.axhline(y=0, color='black', linestyle='--')
        if filename:
            self.save_figure(filename)

    def plot_profiles(self, H, times, pot_minima = None, filename = None):
        """
        Plot height profiles at different times
        Parameters:
            H (ndarray(ndarray)): height profiles
            times (ndarray): plot times
            pot_minima (ndarray): list of minima of binding potential  
        """
        x = self.model.x # get grid of model
        fig, ax = plt.subplots()

        # choose a colormap (e.g. viridis, plasma, cividis, inferno, etc.)
        cmap = cm.viridis  
        norm = mcolors.Normalize(vmin=min(times), vmax=max(times))

        for h, t in zip(H, times):
            color = cmap(norm(t))   # map time to color
            ax.plot(x, h, label=f't={t:.2f}', color = color)
        if pot_minima is not None:
            for y in pot_minima:
                ax.hlines(y, xmin=x[0], xmax=x[-1], linestyles='dashed', color = 'k')

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax = ax)
        cbar.set_label("t")
        ax.set_xlabel('x')
        ax.set_ylabel('h(x,t)')
        #plt.legend(loc = 'right')
        #ax.grid(True)
        if filename:
            self.save_figure(filename)

    def plot_growth(self, H, times, filename = None):
        x = self.model.x
        plt.figure()
        for h, t in zip(H, times):
            growth = self.model.growth_term(h)
            plt.plot(x, growth, label = f"t={t:.2f}")

        plt.xlabel('x')
        plt.ylabel('G(h(x))')
        plt.legend(loc = 'right')
        if filename:
            self.save_figure(filename)

    def plot_free_energy(self, H, times, filename = None):
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
        if filename:
            self.save_figure(filename)