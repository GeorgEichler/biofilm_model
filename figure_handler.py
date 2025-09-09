import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
        #plt.title("Growth function")
        plt.axhline(y=0, color='black', linestyle='--')
        ticks, labels = plt.xticks()
        ticks = list(ticks)
        labels = [lab.get_text() for lab in labels]
        ticks.append(self.model.ha)
        labels.append(r"$h_a$")
        ticks.append(0.2)
        labels.append(r"$h_f$")
        ticks.append(5)
        labels.append(r"$h_{max}$")
        plt.xticks(ticks, labels)
        plt.xlim(0, h_max)

        if filename:
            self.save_figure(filename)

    def plot_profiles(self, H, times, pot_minima = None, plot_filename = None):
        """
        Plot height profiles at different times
        Parameters:
            H (ndarray(ndarray)): height profiles
            times (ndarray): plot times
            pot_minima (ndarray): list of minima of binding potential  
        """
        x = self.model.x # get grid of model

        results = []
        for t, h in zip(times, H):
            for xi, hi in zip(x, h):
                results.append(
                    {
                        "t": t,
                        "h": hi,
                        "x": xi
                    }
                )

        df = pd.DataFrame(results)

        sns.set_theme(style = "white")
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        })
        cmap = sns.color_palette(palette = "crest", as_cmap=True)
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df,
            x="x", y = "h", hue = "t",
            palette=cmap, hue_norm=(times.min(), times.max()),
            ax = ax, legend = False
        )
        

        if pot_minima is not None:
            for y in pot_minima:
                ax.hlines(y, xmin=x[0], xmax=x[-1], linestyles='dashed', color = 'k')

        
        norm = plt.Normalize(times.min(), times.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm = norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = ax)
        cbar.set_ticks(times)
        cbar.set_ticklabels([f"{int(t)}" for t in times])
        cbar.set_label("t")
        
        ax.set_xlabel('x')
        ax.set_ylabel('h(x,t)')
        ax.set_xlim(0, self.model.params['L'])
        ax.set_ylim(bottom = 0)
        #ax.set_xlim(30, 70)
        #plt.legend(loc = 'right')
        #ax.grid(True)
        if plot_filename:
            self.save_figure(plot_filename)

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
            self.save_figure(filename, dpi = 300, bbox_inches = 'thight')

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