from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from gfdm.core import GFDMSolver


def load_config():
    """Load configuration from YAML file"""
    config_path = (
        Path(__file__).parent / "config.yml"
    )  # Get path relative to this script
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert the string to actual numpy expression
    ic_str = config["equation_params"]["initial_condition"]
    config["equation_params"]["initial_condition"] = eval(ic_str, {"np": np})

    return config


def load_points(file_path):
    """Load points from data file"""
    try:
        # Get absolute path relative to current directory
        full_path = Path(__file__).parent.parent / file_path
        # Load points from file
        points = np.loadtxt(full_path)
        return points
    except FileNotFoundError:
        print(f"Error: Could not find data file at {full_path}")
        raise
    except ValueError:
        print(f"Error: Problem reading data from {full_path}. Check file format.")
        raise


def solve_pde(cfg, solver, x, num_steps, inc):
    """Specific code for the PDE of the example
    TODO generalize with config"""

    mu = cfg["equation_params"]["mu"]
    stars = solver.create_stars(x)
    solver.plot_stars(x, stars)

    _, coeffs_for_second_derivative = solver.build_matrices(x, stars)

    # Initialize solutions
    sol = np.ones(len(x)) * cfg["equation_params"]["initial_condition"]
    sol_all_data = np.zeros((num_steps, len(x)))

    # Time derivative loop
    for n in range(num_steps):
        # t = n * inc
        # Exclude 1st and last nodes for the boundary conditions
        for i in range(1, len(x)):
            # Get neighbors from stars
            neighbors = np.where(stars[i] == 1)[0]
            u = (
                sol[neighbors] - sol[i]
            )  # has the size of the number of neighbors and contains distances

            current_u = sol[i]
            # Equation u' = g(u) = d2u/dx2 + mu* u *(1-u)
            second_derivative = coeffs_for_second_derivative[i] @ u
            g_u = second_derivative + mu * current_u * (1 - current_u)

            sol[i] += g_u * inc

        # Boundary conditions (Neumann)
        sol[0] = sol[1]
        sol[-1] = sol[-2]

        # Save step
        sol_all_data[n, :] = sol

    return sol_all_data


def run_example():
    # Load configuration
    cfg = load_config()

    # Extract parameters
    num_steps = cfg["numerical_params"]["num_steps"]
    num_neighbors = cfg["numerical_params"]["num_neighbors"]
    inc = cfg["numerical_params"]["increment"]

    # Create solver
    solver = GFDMSolver(num_neighbors=num_neighbors)

    # Generate points
    x = load_points(cfg["input_data"]["path"])

    # Solve PDE
    sol = solve_pde(cfg, solver, x, num_steps, inc)

    # 3D Plotting
    T = inc * num_steps
    X, T = np.meshgrid(x, np.arange(0, T, inc))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, X, sol, cmap="viridis")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_zlabel("u")
    plt.show()
