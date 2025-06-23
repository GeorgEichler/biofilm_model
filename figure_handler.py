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

        solutions_path = "output"

        for t in timestamps:
            h_temp = h_solutions[t].compute_vertex_values(self.config.mesh)
            time = t * self.config.T / self.config.num_steps
            plt.plot(x, h_temp, label = f'Time = {time}')

        plt.xlabel('x')
        plt.ylabel('h')
        plt.legend()
        plt.title('Height profile')
        if savefig:
            plt.savefig(f"{solutions_path}/simple_height_profile.png")