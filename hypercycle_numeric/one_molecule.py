from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt
from gfdm.core import GFDMSolver


def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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


def solve_pde(cfg, solver, x, num_steps, dt):
    """Solver for one molecule autocatalytic system"""
    # Setup from config
    d = cfg['equation_params']['d']     # diffusion coefficient
    a = cfg['equation_params']['a']     # reaction rate
    p = cfg['equation_params']['p']     # growth exponent
    
    stars = solver.create_stars(x)
    _, coeffs_for_second_derivative = solver.build_matrices(x, stars)

    # Initialize solution
    ic_str = cfg['equation_params']['initial_condition']
    # sol = eval(ic_str, {'np': np, 'x': x})
    sol = eval(ic_str, {'np': np})*np.ones(len(x))
    sol_all_data = np.zeros((num_steps, len(x)))

    print(f"Initial max sol = {np.max(sol)}, min sol = {np.min(sol)}")

    for n in range(num_steps):
        old_sol = sol.copy()
        
        # Calculate f1(t)
        v_powered = (old_sol)**(p+1)  
        f1 = a * np.trapz(v_powered, x)
        
        for i in range(1, len(x)-1):
            neighbors = np.where(stars[i] == 1)[0]
            u = old_sol[neighbors] - old_sol[i]
            
            laplacian = coeffs_for_second_derivative[i] @ u
            
            # More stable reaction term calculation
            v_term = (old_sol[i])**p
            reaction = old_sol[i] * (a * v_term - f1)
            
            # Equation update
            sol[i] = old_sol[i] + dt * (d * laplacian + reaction)
            
            if np.isnan(sol[i]):
                print(f"Step {n}, position {i}:")
                print(f"laplacian={laplacian}, reaction={reaction}")
                print(f"old_sol={old_sol[i]}, f1={f1}")
                return sol_all_data

        # Neumann BCs
        # Should have an extra node on each side! TODO add this
        sol[0] = sol[1]
        sol[-1] = sol[-2]
        
        sol_all_data[n, :] = sol

        # Data checks
        if n % 10 == 0: 
            print(f"Step {n}: max={np.max(sol):.6f}, min={np.min(sol):.6f}, f1={f1:.6f}")

    return sol_all_data


def run_model():
    # Load configuration
    cfg = load_config()

    # Extract parameters
    num_steps = cfg['numerical_params']['num_time_steps']
    num_neighbors = cfg['numerical_params']['num_neighbors']
    inc = cfg['numerical_params']['time_increment']
    
    # Create solver
    solver = GFDMSolver(num_neighbors=num_neighbors)
    
    # Generate points
    x = load_points(cfg['input_data']['path'])

    # Solve PDE
    sol = solve_pde(cfg, solver, x, num_steps, inc)
    t = np.linspace(0, inc * num_steps, num_steps)

    # 3D Plotting
    T = inc * num_steps
    X, T = np.meshgrid(x, np.arange(0, T, inc))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, X, sol, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('v')
    plt.show()
    
    