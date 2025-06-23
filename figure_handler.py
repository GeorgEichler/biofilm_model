import matplotlib.pyplot as plt


class FigureHandler:

    def __init__(self, config):
        self.config = config
        plt.rcParams.update({
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.dpi": 100 #change resolution, standard is 100
        })

    def height_profile(self, h_solutions, timestamps, savefig = False):
        plt.figure()
        x = self.config.mesh.coordinates().flatten()

        for t in timestamps:
            h_temp = h_solutions[t].compute_vertex_values(self.config.mesh)
            plt.plot(x, h_temp, label = f'Time = {t}')

        plt.xlabel('x')
        plt.ylabel('h')
        plt.legend()
        plt.title('Height profile')