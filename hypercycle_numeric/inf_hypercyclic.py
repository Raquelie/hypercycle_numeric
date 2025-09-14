from pathlib import Path

import numpy as np

from gfdm.core import GFDMSolver
from utils.config import load_config
from utils.plotting import plot_3d


def create_points(h, start, end, file_path):
    """
    Create spatial points for the system and save full grid.

    Reads from input file_path and replicates the points
    up to the end of the mesh
    """
    try:
        full_path = Path(__file__).parent.parent / file_path
        base_points = np.loadtxt(full_path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {full_path}")
        raise
    except ValueError:
        print(
            f"Error: Problem reading data from {full_path}. \
              Check file format."
        )
        raise

    n_repetitions = int(np.ceil((end - start) / h))

    full_mesh = []
    base_without_last = base_points[
        :-1
    ]  # Exclude last point of base mesh to avoid duplication
    for i in range(n_repetitions):
        segment = base_without_last + i * h
        full_mesh.extend([x for x in segment if x <= end])

    full_mesh.append(end)  # Add final point
    full_mesh = np.round(np.array(full_mesh), decimals=4)

    output_path = Path(__file__).parent.parent / "data/full_mesh.txt"
    np.savetxt(output_path, full_mesh, fmt="%.4f")  # For reference

    return full_mesh


def get_delayed_x(x, h):
    """Return u(x-h) rounded to nearest mesh point."""
    delayed = x - h
    indices = []

    for d in delayed:
        if d < 0:
            idx = np.abs(x - (d + x[-1])).argmin()
        else:
            idx = np.abs(x - d).argmin()
        indices.append(idx)

    output_path = Path(__file__).parent.parent / "data/delayed_mesh.txt"
    np.savetxt(output_path, x[indices], fmt="%.4f")  # For reference
    return indices


def calculate_f(params, u, x, delayed_indices):
    """Calculate f = int_0^(2π) k(x)*u(x,t)*u(x-h,t) dx."""
    k = params["equation_params"]["k"]
    integrand = k * u * u[delayed_indices]
    integral = np.trapezoid(integrand, x)
    return integral


def solve_pde(cfg, solver, x, num_steps, dt):
    """Solve for continous system."""
    # Setup from config
    alpha = cfg["equation_params"]["alpha"]
    k = cfg["equation_params"]["k"]
    h = cfg["equation_params"]["h"]

    # GFDM
    stars = solver.create_stars(x)
    coeffs_for_first_derivative, coeffs_for_second_derivative = solver.build_matrices(
        x, stars
    )
    # Initialize solution
    ic_str = cfg["equation_params"]["initial_condition"]
    sol = eval(ic_str, {"np": np, "x": x}) * np.ones(len(x))
    sol_all_data = np.zeros((num_steps, len(x)))
    sol_all_data[0, :] = sol
    print(f"Initial max sol = {np.max(sol)}, min sol = {np.min(sol)}")

    # Get delayed indices for fixed h and mesh or use Taylor expansion
    use_taylor_delay = cfg["numerical_params"].get("use_taylor_delay", False)
    if use_taylor_delay:
        delayed_indices = None  # Not used in Taylor mode
    else:
        delayed_indices = get_delayed_x(x, h)

    print("\nSolving the system...")

    for n in range(1, num_steps):
        current_u = sol.copy()
        # Calculate derivatives once per time step
        u_x = np.zeros_like(current_u)
        u_xx = np.zeros_like(current_u)
        for i in range(1, len(x) - 1):
            neighbors = np.where(stars[i] == 1)[0]
            u_x[i] = coeffs_for_first_derivative[i] @ (
                current_u[neighbors] - current_u[i]
            )
            u_xx[i] = coeffs_for_second_derivative[i] @ (
                current_u[neighbors] - current_u[i]
            )

        if use_taylor_delay:
            # Use Taylor expansion for delay
            delayed_u = current_u - h * u_x + 0.5 * h**2 * u_xx
            k = cfg["equation_params"]["k"]
            integrand = k * current_u * delayed_u
            f = np.trapezoid(integrand, x)
        else:
            f = calculate_f(cfg, current_u, x, delayed_indices)
            delayed_u = current_u[delayed_indices]

        for i in range(1, len(x) - 1):
            neighbors = np.where(stars[i] == 1)[0]
            # Use already computed second derivative
            laplacian = u_xx[i]
            diff = alpha * laplacian
            # Reaction: k*u(x-h)-f
            delayed_term = k * delayed_u[i]
            reaction = delayed_term - f
            # Equation update
            sol[i] = current_u[i] + dt * (current_u[i] * reaction + diff)

            if np.isnan(sol[i]):
                print(f"Step {n}, position {i}:")
                print(f"laplacian={laplacian}, reaction={reaction}")
                print(f"sol={current_u[i]}, f={f}")
                return sol_all_data

        # Neumann BCs
        sol[-1] = 0.5 * (sol[1] + sol[-2])
        # Dirichlet BCs
        sol[0] = sol[-1]

        sol_all_data[n, :] = sol

        # Data checks, because it tends to go to infinity :(
        if n % 1000 == 0:
            print(
                f"Step {n}: max={np.max(sol):.6f}, \
                  min={np.min(sol):.6f}, f={f:.6f}"
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
    print("INFINITE HYPERCYCLE MODEL SIMULATION")
    print("=" * 50 + "\n")

    # Load configuration
    config_path = Path(__file__).parent / "config_inf.yml"
    cfg = load_config(config_path)

    # Extract parameters
    num_steps = cfg["numerical_params"]["num_time_steps"]
    num_neighbors = cfg["numerical_params"]["num_neighbors"]
    delta_t = cfg["numerical_params"]["time_increment"]
    h = cfg["equation_params"]["h"]
    start = cfg["equation_params"]["start"]
    end = cfg["equation_params"]["end"]
    path = cfg["input_data"]["path"]

    print("\nModel Parameters:")
    print("-" * 20)
    print(f"Time steps: {num_steps}")
    print(f"Number of neighbors: {num_neighbors}")
    print(f"Time increment: {delta_t}")
    print(f"Catalyzer delay: {h}")
    print(f"Domain: [{start}, {end}]")
    print(f"Input data path: {path}\n")

    # Create solver
    solver = GFDMSolver(num_neighbors=num_neighbors)

    # Generate points
    x = create_points(h, start, end, path)

    # Solve PDE
    sol = solve_pde(cfg, solver, x, num_steps, delta_t)

    # 3D Plotting
    plot_3d(num_steps, delta_t, x, sol, "inf_model")

    print("\n" + "=" * 50)
    print("Simulation completed successfully")
    print("=" * 50 + "\n")
