# Autocathalitic system 

## Usage
- Check `hypercycle_numeric/config_n.yml` for parameters. The number of molecules needs to be added as well as the constant for each of these molecules
- Code is in `hypercycle_numeric/n_mol_hypercyclic.py`
- Run with main.py after checking the call to the function is in the file and uncommented

## Description
The example aims to create a numerical simulation of the model proposed by:

> Bratus, A. S., Posvyanskii, V. P., & Novozhilov, A. S. (2009). "Existence and stability of stationary solutions to spatially extended autocatalytic and hypercyclic systems under global regulation and with nonlinear growth rates." *arXiv preprint*, arXiv:0901.3556v1. [DOI: 10.48550/arXiv.0901.3556](https://doi.org/10.48550/arXiv.0901.3556)

With n molecules, the hypercyclic model has the form:

$\partial_t v_i = v_i(av_{i-1}^m - f_2(t)) + d_i\Delta v_i, \quad i=1,\ldots,n, \quad t>0$

where:

$f_2(t) = \sum_{i=1}^n \int_\Omega a_iv_{i-1}^m(x,t)v_i(x,t)\,dx$

And Neumann's boundary conditions.

For the spatial derivative the GFDM method is used, where the input mesh is taken from the `data` folder according to the config. For the integral, numpy's trapezoid rule is used.

## Output
A plot of the solution (t, x, v) is generated for the n molecules.

