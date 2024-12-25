# Generalized Finite Differences Method examples

## Example 1

### Usage
- Check `config.yml` for parameters
- Run with main.py

### Description
The example solves $u' = g(u) = \frac{\partial^2 u}{\partial x^2} + \mu u(1-u)$ with an initial condition and Neumann boundary conditions:
$$ \frac{\partial u}{\partial x}\bigg|_{x=0} = 0, \quad \frac{\partial u}{\partial x}\bigg|_{x=L} = 0 $$

For the GFDM method, the input mesh is take from the `data` folder.

### Output
A plot for the stars in generated in the `output`folder. 
A plot of the solution is generated.


