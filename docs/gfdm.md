# GDFM module

The folder `gdfm` contains an implementation of the GFDM method for integer derivatives as described in:

> Vargas, A. M. (2022). "Finite difference method for solving fractional differential equations at irregular meshes." *Mathematics and Computers in Simulation*, 193, 204-216. [DOI: 10.1016/j.matcom.2021.10.010](https://doi.org/10.1016/j.matcom.2021.10.010)


## Description


```
gfdm/
|-- __init__.py
|-- core.py
|-- utils.py
`-- weights.py

```

- `core.py`: Solver class
- `utils.py`: Implementation of the algorithm
- `weights.py`: Different weight functions for the method

## Usage

Examples can be found in the `examples` folder, and documentation for them under `docs/gdfm_examples.md`

For creating a new example, a solver object has to be created:

```
from gfdm.core import GFDMSolver
solver = GFDMSolver(num_neighbors=num_neighbors)
```

The methods `create_stars`, `plot_stars` and `build_matrices` can then be used on the object created.
