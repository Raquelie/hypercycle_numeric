# Generalized Finite Differences Method examples

## Example 1

### Usage
- Check `config.yml` for parameters
- Run with main.py

### Description
The example solves $u' = g(u) = \frac{\partial^2 u}{\partial x^2} + \mu u(1-u)$ with an initial condition and Neumann boundary conditions:

$u_x(0) = 0, \quad u_x(L) = 0$

For the GFDM method, the input mesh is taken from the `data` folder.

### Output
A plot for the stars in generated in the `output`folder. 
A plot of the solution is generated.


## Example 2

### Usage
- Check `config.yml` for parameters
- Run with main.py

### Description
The example solves $u' = g(u) = \frac{\partial^2 u}{\partial x^2} + \mu u(1-u) + x² -2 -\mu x^2 (1- x² e^t)$ with an initial condition and Dirichlet boundary conditions.

For the GFDM method, the input mesh is taken from the `data` folder.

### Output
A plot for the stars in generated in the `output` folder. 
A plot of the solution is generated along with the exact solution $u = x² e^t$.


