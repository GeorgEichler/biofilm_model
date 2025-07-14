# Thin-film equation model for biofilms
The goal of this project is to formulate a thin-film equations model simulating the emergence of biofilm layers and comparing the results to the 
experimental observations from [Dhar et al.](https://www.nature.com/articles/s41567-022-01641-9).
Using methods discussed in [Yin et al.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.023104), we simulate the height of the biofilm using a thin-film equation of the form
<p align=center>
$\frac{\partial h}{\partial t} = \nabla \cdot \left(\cdot \nabla (-\gamma \nabla^2 h - \Pi(h)) \right) + g (h - h_0) (1 - (h-h_0)/h_{max}),$
</p>

which is already non-dimensionalised. Here $h=h(t,\mathbf{x})$ is the height, $g$ is the ratio of growth and diffusion rate, $h_0$ is the height of the precursor film, $h_{max}$ is a limiting maximal height and $\Pi(h)$ is the disjoining pressure given as
<p align=center>
$\Pi_1(h) = a e^{-h/c} ( k \sin(hk + b) + 1/c \cdot  \cos(hk + b)) + \frac{d}{2c}e^{-h/(2c)}$
</p>
or
<p align=center>
$\Pi_2(h) = a e^{-h/c} ( k \sin(hk + b) + 1/c \cdot  \cos(hk + b)) + \frac{2d}{c}e^{-(2h)/c}$
</p>

and corresponds to a binding potential $g$ given by $\Pi = - \partial g/\partial h$.
