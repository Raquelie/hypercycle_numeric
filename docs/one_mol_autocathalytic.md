# Autocathalitic system

## Usage
- Check `hypercycle_numeric/config.yml` for parameters
- Code is in `hypercycle_numeric/one_mol_autocathalitic.py`
- Run with main.py using the parameter `one_mol_auto`

## Description
The example aims to create a numerical simulation of the model proposed by:

> Bratus, A. S., Posvyanskii, V. P., & Novozhilov, A. S. (2009). "Existence and stability of stationary solutions to spatially extended autocatalytic and hypercyclic systems under global regulation and with nonlinear growth rates." *arXiv preprint*, arXiv:0901.3556v1. [DOI: 10.48550/arXiv.0901.3556](https://doi.org/10.48550/arXiv.0901.3556)

With one molecule, the autocathalitic model has the form:

$\partial_t v_1 = v_1(a_1 v_1^p - f_1(t)) + d_1 \Delta v_1$

where:

$f_1(t) = \int_\Omega a_1 v_1^{p+1}(x,t) \, dx$

And Neumann's boundary conditions.

For the spatial derivative the GFDM method is used, where the input mesh is taken from the `data` folder according to the config. For the integral, numpy's trapezoid rule is used.

## Output
A plot of the solution (t, x, v) is generated in the `output` folder.
