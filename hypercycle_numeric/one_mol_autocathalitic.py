from pathlib import Path

import numpy as np

from gfdm.core import GFDMSolver
from utils.config import load_config
from utils.plotting import plot_3d


def load_points(file_path):
    """Load spatial points from data file."""
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


def solve_pde(cfg, solver, x, num_steps, dt):
    """Solve for one molecule autocatalytic system."""
    # Setup from config
    d = cfg["equation_params"]["d"]  # diffusion coefficient
    a = cfg["equation_params"]["a"]  # reaction rate
    p = cfg["equation_params"]["p"]  # growth exponent

    stars = solver.create_stars(x)
    _, coeffs_for_second_derivative = solver.build_matrices(x, stars)

    # Initialize solution
    ic_str = cfg["equation_params"]["initial_condition"]
    sol = eval(ic_str, {"np": np, "x": x}) * np.ones(len(x))
    sol_all_data = np.zeros((num_steps, len(x)))
    sol_all_data[0, :] = sol
    print(f"Initial max sol = {np.max(sol)}, min sol = {np.min(sol)}")

    print("\nSolving the system...")

    for n in range(1, num_steps):
        current_v = sol.copy()
        # Calculate f1(t) = int(a*v^(p+1)*dx) for x in domain
        # This is a global term
        v_powered = (current_v) ** (p + 1)
        # Use numpy's trapezoid rule, a is constant
        f1 = a * np.trapezoid(v_powered, x)

        for i in range(1, len(x) - 1):
            neighbors = np.where(stars[i] == 1)[0]
            u = current_v[neighbors] - current_v[i]
            laplacian = coeffs_for_second_derivative[i] @ u
            # Reaction term plus global regulation: v*(a*v^p - f1)
            v_term = (current_v[i]) ** p
            reaction = current_v[i] * (a * v_term - f1)
            # Equation update
            sol[i] = current_v[i] + dt * (d * laplacian + reaction)

            if np.isnan(sol[i]):
                print(f"Step {n}, position {i}:")
                print(f"laplacian={laplacian}, reaction={reaction}")
                print(f"sol={current_v[i]}, f1={f1}")
                return sol_all_data

        # Neumann BCs
        # Should have an extra node on each side! TODO: add this
        sol[0] = sol[1]
        sol[-1] = sol[-2]

        sol_all_data[n, :] = sol

        # Data checks, because it tends to go to infinity :(
        if n % 100 == 0:
            print(
                f"Step {n}: max={np.max(sol):.6f}, min={np.min(sol):.6f}, f1={f1:.6f}"
            )

    return sol_all_data


def run_model():
    """
    Run the one molecule auto cathalytic model and visualize results.

    Steps:
    1. Load configuration parameters
    2. Initialize GFDM solver
    3. Generate spatial points
    4. Solve integro-differential system
    5. Create 3D visualization
    """
    print("\n" + "=" * 50)
    print("ONE MOLECULE AUTO CATHALYTIC MODEL SIMULATION")
    print("=" * 50 + "\n")

    # Load configuration
    config_path = Path(__file__).parent / "config.yml"
    cfg = load_config(config_path)

    # Extract parameters
    num_steps = cfg["numerical_params"]["num_time_steps"]
    num_neighbors = cfg["numerical_params"]["num_neighbors"]
    delta_t = cfg["numerical_params"]["time_increment"]
    input_data_path = cfg["input_data"]["path"]

    print("\nModel Parameters:")
    print("-" * 20)
    print(f"Time steps: {num_steps}")
    print(f"Number of neighbors: {num_neighbors}")
    print(f"Time increment: {delta_t}")
    print(f"Input data path: {input_data_path}\n")

    # Create solver
    solver = GFDMSolver(num_neighbors=num_neighbors)

    # Generate points
    x = load_points(input_data_path)

    # Solve PDE
    sol = solve_pde(cfg, solver, x, num_steps, delta_t)

    # 3D Plotting
    plot_3d(num_steps, delta_t, x, sol, "one_mol")

    print("\n" + "=" * 50)
    print("Simulation completed successfully")
    print("=" * 50 + "\n")
