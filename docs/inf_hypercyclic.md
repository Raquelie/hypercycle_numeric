# Hypercyclic system, spatially explicit, with n molecules

## Usage
- Check `hypercycle_numeric/config_inf.yml` for parameters.
- Code is in `hypercycle_numeric/inf_hypercyclic.py`
- Run with main.py using the parameter `inf_hyper`

## Description
The example aims to create a numerical simulation of the model proposed by:

> A.S. Bratus, O.S. Chmereva, I. Yegorov, A.S. Novozhilov, "On a hypercycle equation with infinitely many members," Journal of Mathematical Analysis and Applications, Volume 521, Issue 2, 15 May 2023, 126988. https://doi.org/10.1016/j.jmaa.2022.126988


The system from this paper is:

$$
\begin{cases}
\frac{\partial u(x,t)}{\partial t} = u(x,t)(k(x)u(x-h,t) - f[u(\cdot,t)]) + \alpha \frac{\partial^2 u(x,t)}{\partial x^2}, \\
u(x-h,t) = u(x-h+2\pi,t) \quad \text{if} \quad -h \leq x-h < 0, \\
u(x,0) = \varphi(x) \quad \text{(initial condition)}, \\
u(0,t) = u(2\pi,t), \quad \frac{\partial u}{\partial x}(0,t) = \frac{\partial u}{\partial x}(2\pi,t) \quad \text{(boundary conditions)}, \\
0 \leq x \leq 2\pi, \quad t \geq 0.
\end{cases}
$$

where:

$f[u(\cdot,t)] = \int_0^{2\pi} k(x)u(x,t)u(x-h,t)\,dx \quad \forall t \geq 0$

For the spatial derivative the GFDM method is used, where the input mesh is taken from the `data` folder according to the config. For the integral, numpy's trapezoid rule is used.

## Output
A plot of the solution (t, x, v) is generated for the system in the `output` folder.
