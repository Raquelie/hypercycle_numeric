from pathlib import Path
import numpy as np
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from gfdm.core import GFDMSolver


def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config_inf.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    

def create_points(h, start, end, file_path):
    """Create spatial points for the system and save full grid"""
    try:
        full_path = Path(__file__).parent.parent / file_path
        base_points = np.loadtxt(full_path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {full_path}")
        raise
    except ValueError:
        print(f"Error: Problem reading data from {full_path}. Check file format.")
        raise

    n_repetitions = int(np.ceil((end - start) / h))
    
    full_mesh = []
    base_without_last = base_points[:-1]  # Exclude last point of base mesh to avoid duplication
    for i in range(n_repetitions):
        segment = base_without_last + i*h
        full_mesh.extend([x for x in segment if x <= end])
    
    full_mesh.append(end)  # Add final point
    full_mesh = np.round(np.array(full_mesh), decimals=4)
    
    output_path = Path(__file__).parent.parent / 'data/full_mesh.txt'
    np.savetxt(output_path, full_mesh, fmt='%.4f') # For reference
    
    return full_mesh


def get_delayed_u(u, x, h):
    """Returns u(x-h) rounded to nearest mesh point"""
    delayed = x - h
    delayed = np.where(delayed < 0, delayed + x[-1], delayed)
    indices = np.array([np.abs(x - d).argmin() for d in delayed[:-1]])  # Exclude last point
    indices = np.append(indices, indices[0])  # Add first index for last point
    return u[indices]


def calculate_f(params, u, x):
    """Calculate f = int_0^(2π) k(x)*u(x,t)*u(x-h,t) dx"""
    k = params['equation_params']['k']
    h = params['equation_params']['h']
    delayed_u = get_delayed_u(u, x, h)
    integrand = k * u * delayed_u
    integral = np.trapezoid(integrand, x)
    return integral


def solve_pde(cfg, solver, x, num_steps, dt):
    """Solver for continous system"""
   
    # Setup from config
    alpha = cfg['equation_params']['alpha']
    k = cfg['equation_params']['k']
    h = cfg['equation_params']['h']
    
    # GFDM
    stars = solver.create_stars(x)
    _, coeffs_for_second_derivative = solver.build_matrices(x, stars)

    # Initialize solution
    ic_str = cfg['equation_params']['initial_condition']
    sol = eval(ic_str, {'np': np, 'x': x})*np.ones(len(x))
    sol_all_data = np.zeros((num_steps, len(x)))
    sol_all_data[0, :] = sol
    print(f"Initial max sol = {np.max(sol)}, min sol = {np.min(sol)}")

    for n in range(1, num_steps):
        current_u = sol.copy()
        # This is a global term 
        f = calculate_f(cfg, current_u, x)
        delayed_u = get_delayed_u(current_u,x,h)
        for i in range(1, len(x)-1):
            neighbors = np.where(stars[i] == 1)[0]
            u = current_u[neighbors] - current_u[i]
            laplacian = coeffs_for_second_derivative[i] @ u
            # Diffusion: alpha * laplacian
            diff = alpha*laplacian
            # Reaction: k*u(x-h)-f
            delayed_term = k*delayed_u[i]
            reaction = delayed_term - f
            # Equation update
            sol[i] = current_u[i] + dt * (current_u[i]*reaction + diff)

            if np.isnan(sol[i]):
                print(f"Step {n}, position {i}:")
                print(f"laplacian={laplacian}, reaction={reaction}")
                print(f"sol={current_u[i]}, f={f}")
                return sol_all_data
        
        # Dirichlet BCs
        sol[0] = sol[-1]
        # Neumann BCs
        sol[1] = sol[0]
        sol[-2] = sol[-1]

        sol_all_data[n, :] = sol

        # Data checks, because it tends to go to infinity :(
        if n % 1 == 0: 
            print(f"Step {n}: max={np.max(sol):.6f}, min={np.min(sol):.6f}, f={f:.6f}")
    return sol_all_data


def run_model():
    """Run the model and plot the solution"""
    # Load configuration
    cfg = load_config()

    # Extract parameters
    num_steps = cfg['numerical_params']['num_time_steps']
    num_neighbors = cfg['numerical_params']['num_neighbors']
    inc = cfg['numerical_params']['time_increment']
    h = cfg['equation_params']['h']
    start = cfg['equation_params']['start']
    end = cfg['equation_params']['end']
    path = cfg['input_data']['path']

    # Create solver
    solver = GFDMSolver(num_neighbors=num_neighbors)
    
    # Generate points
    x = create_points(h, start, end, path)

    # Solve PDE
    sol = solve_pde(cfg, solver, x, num_steps, inc)

    # 3D Plotting
    T = inc * num_steps
    X, T = np.meshgrid(x, np.arange(0, T, inc))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, sol, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('v')
    ax.set_xlim(np.max(T), 0)  # Reverse t axis direction
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plot
    plt.savefig(output_dir / f"inf_model_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    


    
    