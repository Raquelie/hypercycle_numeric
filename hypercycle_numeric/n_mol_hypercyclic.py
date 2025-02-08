from pathlib import Path

import numpy as np

from gfdm.core import GFDMSolver
from utils.config import load_config
from utils.plotting import plot_multiple


def load_points(file_path):
    """Load points from data file."""
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


def calculate_f2(n_molecules, params, v, x, p):
    """Calculate f2 = sum_i(int_x(a_i * v_{i-1}^p * v_i * dx)))."""
    f2 = 0.0
    for i in range(1, n_molecules + 1):
        try:
            prev_v = v[i - 2]
        except IndexError:
            prev_v = v[n_molecules - 1]
        integrand = params[i]["a"] * v[i - 1] * prev_v**p
        integral = np.trapezoid(integrand, x)
        f2 = f2 + integral
    return f2


def solve_pde(cfg, solver, x, num_steps, dt):
    """Solve for one molecule autocatalytic system."""
    # Setup from config
    eq_params = {}
    n_molecules = cfg["equation_params"]["number_of_molecules"]
    p = cfg["equation_params"]["p"]

    for i in range(1, n_molecules + 1):
        molecule_key = f"molecule_{i}"
        if molecule_key in cfg["equation_params"]:
            eq_params[i] = {
                "d": cfg["equation_params"][molecule_key]["d"],
                "a": cfg["equation_params"][molecule_key]["a"],
                "initial_condition": cfg["equation_params"][molecule_key][
                    "initial_condition"
                ],
            }
    # GFDM
    stars = solver.create_stars(x)
    _, coeffs_for_second_derivative = solver.build_matrices(x, stars)

    # Initialize arrays
    sol = np.zeros((n_molecules, len(x)))
    sol_all_data = np.zeros(
        (n_molecules, num_steps, len(x))
    )  # has one more dimension for the n molecules

    # Set up initial conditions for each molecule
    for i in range(1, n_molecules + 1):
        ic_str = eq_params[i]["initial_condition"]
        sol[i - 1] = eval(ic_str, {"np": np, "x": x}) * np.ones(len(x))
        sol_all_data[i - 1, 0] = sol[i - 1]  # Set initial condition in solution tensor

    print(f"Initial max sol = {np.max(sol)}, min sol = {np.min(sol)}")

    print("\nSolving the system...")

    for n in range(1, num_steps):
        current_v = sol.copy()

        # Calculate f2(t) = sum(int(a*v^(p+1)*dx))
        # for x in domain, n in molecules for each t
        # This is a global term over all molecules in the system
        f2 = calculate_f2(n_molecules, eq_params, current_v, x, p)

        # Spatial loop in i
        for i in range(1, len(x) - 1):
            neighbors = np.where(stars[i] == 1)[0]
            u = np.zeros((n_molecules, cfg["numerical_params"]["num_neighbors"]))
            laplacian_at_x = np.zeros(n_molecules)
            v_term_at_x = np.zeros(n_molecules)
            diffusion = np.zeros(n_molecules)
            reaction = np.zeros(n_molecules)
            # Terms that need iteration over the molecules
            for m in range(n_molecules):
                # second spatial derivative
                u[m] = current_v[m][neighbors] - current_v[m][i]
                laplacian_at_x[m] = coeffs_for_second_derivative[i] @ u[m]
                # v term which couples the molecules
                try:
                    v_term_at_x[m] = (current_v[m - 1][i]) ** p
                except IndexError:
                    v_term_at_x[m] = (current_v[n_molecules - 1][i]) ** p

                # Diffusion array, one value for each molecule
                diffusion[m] = eq_params[m + 1]["d"] * laplacian_at_x[m]

                # Reaction term plus global regulation:
                # v_m*(a_m*v_{m-1}^p - f2) with m as the molecule index
                reaction[m] = current_v[m, i] * (
                    eq_params[m + 1]["a"] * v_term_at_x[m] - f2
                )

            # Equation update, all vectors of size n_molecules
            # except for dt which is a scalar
            sol[:, i] = current_v[:, i] + dt * (diffusion + reaction)

            if np.isnan(sol[:, i].any()):
                print(f"Step {n}, position {i}:")
                print(f"diffusion={diffusion}, reaction={reaction}")
                print(f"sol={current_v[:, i]}, f2={f2}")
                return sol_all_data

        # Neumann BCs
        # Should have an extra node on each side! TODO: add this
        sol[:, 0] = sol[:, 1]
        sol[:, -1] = sol[:, -2]

        sol_all_data[:, n, :] = sol

        # Data checks, because it tends to go to infinity :(
        if n % 1000 == 0:
            for i in range(n_molecules):
                v_max = np.max(sol[i])
                v_min = np.min(sol[i])
                print(
                    f"Step {n}, Molecule {i+1}: max={v_max:.6f}, \
                        min={v_min:.6f}, f2={f2:.6f}"
                )

    return sol_all_data


def run_model():
    """
    Run the infinite hypercycle model and visualize results.

    Steps:
    1. Load configuration parameters
    2. Initialize GFDM solver
    3. Generate spatial points
    4. Solve integro-differential system
    5. Create 3D visualization
    """
    print("\n" + "=" * 50)
    print("N MOLECULE HYPERCYCLE MODEL SIMULATION")
    print("=" * 50 + "\n")

    # Load configuration
    config_path = Path(__file__).parent / "config_n.yml"
    cfg = load_config(config_path)

    # Extract parameters
    num_steps = cfg["numerical_params"]["num_time_steps"]
    num_neighbors = cfg["numerical_params"]["num_neighbors"]
    delta_t = cfg["numerical_params"]["time_increment"]
    num_molecules = cfg["equation_params"]["number_of_molecules"]
    input_data_path = cfg["input_data"]["path"]

    print("\nModel Parameters:")
    print("-" * 20)
    print(f"Time steps: {num_steps}")
    print(f"Number of neighbors: {num_neighbors}")
    print(f"Time increment: {delta_t}")
    print(f"Num molecules: {num_molecules}")
    print(f"Input data path: {input_data_path}\n")

    # Create solver
    solver = GFDMSolver(num_neighbors=num_neighbors)

    # Generate points
    x = load_points(input_data_path)

    # Solve PDE
    sol = solve_pde(cfg, solver, x, num_steps, delta_t)

    # 3D Plotting
    plot_multiple(num_steps, delta_t, x, num_molecules, sol)

    print("\n" + "=" * 50)
    print("Simulation completed successfully")
    print("=" * 50 + "\n")
