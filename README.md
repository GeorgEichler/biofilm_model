# A thin-film model for the formation of biofilms
This goal of this repository is to formulate a thin-film model simulating the development of a few layers of biofilms and analyse the
the time evolution with regard to its parameter values. Based on ideas discussed in [Yin et al.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.023104) and [Pietz et al.](https://pubs.rsc.org/en/content/articlehtml/2025/sm/d4sm01463d), we formulate a thin-film equation of the form
<p align=center>
$\frac{\partial h}{\partial t} = \nabla \cdot \left( \nabla ( \nabla^2 h - \epsilon \Pi(h)) \right) + g (h - h_a) (1 - (h)/h_{max}) * (1 - \exp(h_f - h),$
</p>

which is already non-dimensionalised. Here $h=h(t,\mathbf{x})$ is the height of the profile, $g$ is the ratio of growth rate, $h_a$ is an activation height to only allow one layer to grow further, $h_f$ being a stabilising fixed point for the precursor height, $h_{max}$ is a limiting maximal height and $\Pi(h)$ is the disjoining pressure given as
<p align=center>
$\Pi(h) = a e^{-h/c} ( k \sin(hk + b) + 1/c \cdot  \cos(hk + b)) + \frac{d}{e}e^{-h/(e)}$
</p>
and is derived from the binding potential
<p align=center>
$f(h) = a \cos(hk+b)e^{-h/c} + d e^{-h/e}$
</p>

and its relation to the binding potential is given via $\Pi = - \partial g/\partial h$. Thereby, we interpret the first minimum of the binding potential as the precursor height and all the latter ones as as the first, second and so on minima.
We analyse the time evolution of the thin-film equation for the width, maximal height and the time it takes until a second layer is formed with regard to the energy scale parameter $\epsilon$ and the growth rate $g$.


