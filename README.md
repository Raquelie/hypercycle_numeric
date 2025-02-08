# A study of spatial hypercyclic models using the generalized finite differences method

## Summary

Hypercycles are finite closed networks of self-replicating molecules [1]. While traditional models for hypercycles use a system of ODEs, newer research [1] [2] has shown that spatially explicit models using PDEs have an intersting behaviour. In order to numerically solve these systems, the Generalized Finite Differences Method (GFDM) [3] is implemented to compute spatial derivatives.

## Usage

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

See docs folder for specifics of the available models.

- GDFM method examples `example_1` and `example_2`: `docs/gdfm_examples.md`
- One molecule auto cathalytic model `one_mol_auto`: `docs/one_mol_autocathalytic.md`
- N-molecule hypercyclic model `n_mol_hyper`: `docs/n_mol_hypercylic.md`
- Continous hypercyclic model `inf_hyper`: `docs/inf_hypercyclic.md`

When ready, use:

`python main.py <model_to_run>`

Replace `<model_to_run>` with one of the available models:

- `example_1`
- `example_2`
- `one_mol_auto`
- `n_mol_hyper`
- `inf_hyper`

## Development

### Pre-commit hooks

This repository uses pre-commit hooks to ensure code quality. To set up pre-commit in your local environment:

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Install the git hooks:
```bash
pre-commit install
```

The pre-commit hooks will run automatically on `git commit`. They include:
- Black (code formatting)
- Flake8 (code style)
- isort (import sorting)

## References

[1] Alexander S. Bratus, Olga S. Chmereva, Ivan Yegorov, and Artem S. Novozhilov. "On a hypercycle equation with infinitely many members." Journal of Mathematical Analysis and Applications, vol. 521, no. 2, 2023, p. 126988. doi: https://doi.org/10.1016/j.jmaa.2022.126988

[2] Alexander S. Bratus', Vladimir P. Posvyanskii, and Artem S. Novozhilov. "Existence and stability of stationary solutions to spatially extended autocatalytic and hypercyclic systems under global regulation and with nonlinear growth rates." arXiv preprint arXiv:0901.3556, 2009.

[3] Antonio M. Vargas. "Finite difference method for solving fractional differential equations at irregular meshes." Mathematics and Computers in Simulation, vol. 193, 2022, pp. 204-216. doi: https://doi.org/10.1016/j.matcom.2021.10.010
