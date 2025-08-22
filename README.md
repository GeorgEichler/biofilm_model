# Thin-film equation model for biofilms
The goal of this project is to formulate a thin-film equations model simulating the emergence of biofilm layers and comparing the results to the 
experimental observations from [Dhar et al.](https://www.nature.com/articles/s41567-022-01641-9).
Using methods discussed in [Yin et al.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.023104), we simulate the height of the biofilm using a thin-film equation of the form
<p align=center>
$\frac{\partial h}{\partial t} = \nabla \cdot \left(\cdot \nabla ( \nabla^2 h - \epsilon \Pi(h)) \right) + g (h - h_a) (1 - (h)/h_{max}) * (1 - \exp(h_f - h),$
</p>

which is already non-dimensionalised. Here $h=h(t,\mathbf{x})$ is the height of the profile, $g$ is the ratio of growth rate, $h_a$ is an activation height to only allow one layer to grow further, $h_f$ being a stabilising fixed point for the precursor height, $h_{max}$ is a limiting maximal height and $\Pi(h)$ is the disjoining pressure given as
<p align=center>
$\Pi(h) = a e^{-h/c} ( k \sin(hk + b) + 1/c \cdot  \cos(hk + b)) + \frac{d}{e}e^{-h/(e)}$
</p>
and is derived from the binding potential
<p align=center>
$f(h) = a \cos(hk+b)e^{-h/c} + d e^{-h/e}$
</p>

and its relation to the binding potential is given via $\Pi = - \partial g/\partial h$.
This repository aims to examine the evolution of the binding film and explore transition points for different parameter sets as well as determine the speed of biofilm spreading.
