import numpy as np
import matplotlib.pyplot as plt

class FigureHandler:
    """
    Handling of plots for the thin-film equation model
    """
    def __init__(self, model):
        self.model = model

    def plot_binding_energy(self, g, h_min = 0, h_max = 10, nh = 1001):
        h_array = np.linspace(h_min, h_max, nh)
        plt.figure()
        plt.plot(h_array, g(h_array))
        plt.xlabel('h')
        plt.ylabel('g(h)')
        plt.title('Binding potential')

    def plot_profiles(self, H, times):
        x = self.model.x # get grid of model
        plt.figure()
        for h, t in zip(H.T, times):
            plt.plot(x, h, label=f't={t:.2f}')
        plt.xlabel('x')
        plt.ylabel('h(x,t)')
        plt.legend()
        plt.grid(True)