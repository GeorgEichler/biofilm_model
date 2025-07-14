# Thin-film equation model for biofilms
The goal of this project is to formulate a thin-film equations model simulating the emergence of biofilm layers and comparing the results to the 
experimental observations from [Dhar et al.](https://www.nature.com/articles/s41567-022-01641-9).
Using methods discussed in [Yin et al.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.023104), we simulate the height of the biofilm using a thin-film equation of the form
$$
\frac{\partial h}{\partial t} = \nabla \cdot \left[Q \cdot \nabla (-\gamma \nabla^2 h - \Pi(h)) \right] + g (h - h_0) (1 - (h-h_0)/h_{max})
$$
