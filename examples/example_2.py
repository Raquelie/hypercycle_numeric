from datetime import datetime
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


def exact_solution(x, t):
    return x**2 * np.exp(t)


def forcing_term(x, t, mu):
    return x**2 - 2 - mu * x**2 * (1 - x**2 * np.exp(t))


def solve_pde(cfg, solver, x, num_pasos, inc):
    """Specific code for the PDE of the example
    TODO generalize with config"""

    mu = cfg["equation_params"]["mu"]
    stars = solver.create_stars(x)
    solver.plot_stars(x, stars)

    _, coeffs_for_second_derivative = solver.build_matrices(x, stars)

    # Initialize solutions
    # Use the initial condition of the exact solution
    sol = np.array([x_i**2 for x_i in x])  # Initial condition u(x,0) = x²
    sol_all_data = np.zeros((num_pasos, len(x)))

    # Time derivative loop
    for n in range(num_pasos):
        t = n * inc
        # Exclude first and last for BC
        for i in range(1, len(x)):
            # Get neighbors from stars
            neighbors = np.where(stars[i] == 1)[0]
            u = (
                sol[neighbors] - sol[i]
            )  # has the size of the number of neighbors and contains distances

            current_u = sol[i]
            # Equation u' = g(u) = d2u/dx2 + mu* u *(1-u) + f
            f = forcing_term(x[i], t, mu)
            second_derivative = coeffs_for_second_derivative[i] @ u
            g_u = second_derivative + mu * current_u * (1 - current_u) + f
            sol[i] += g_u * inc

        # Boundary conditions (Dirichlet)
        sol[0] = exact_solution(0, t)
        sol[-1] = exact_solution(1, t)

        # Save step
        sol_all_data[n, :] = sol

    return sol_all_data


def run_example():
    # Load configuration
    cfg = load_config()

    # Extract parameters
    num_steps = cfg["numerical_params"]["num_steps"]
    num_neighbors = cfg["numerical_params"]["num_neighbors"]
    delta_t = cfg["numerical_params"]["increment"]

    # Create solver
    solver = GFDMSolver(num_neighbors=num_neighbors)

    # Generate points
    x = load_points(cfg["input_data"]["path"])

    # Solve PDE
    sol = solve_pde(cfg, solver, x, num_steps, delta_t)

    # 2D Plotting
    t_final = num_steps * delta_t
    exact = exact_solution(x, t_final)
    plt.plot(x, sol[-1, :], "b-", label="Numerical")
    plt.plot(x, exact, "r--", label="Exact")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save plot
    plt.savefig(output_dir / f"function_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()
