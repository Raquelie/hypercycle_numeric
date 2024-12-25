import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path


def create_stars(x, num_neighbors=2):
    """Create stars for each point in x"""
    num_points = len(x)
    stars = np.zeros((num_points, num_points), dtype=int)

    for i in range(num_points):
        distances = np.abs(x - x[i])
        distances[i] = np.inf
        neighbors = np.argpartition(distances, num_neighbors)[:num_neighbors]
        stars[i, neighbors] = 1
    
    return stars


def build_matrices(x, stars, internal_nodes, weight_function):
    """
    Build coefficientes for first and second derivatives
    
    Parameters:
    -----------
    x : array_like
        Points coordinates
    stars : array_like
        Binary matrix where stars[i,j] = 1 if j is in star of i
    internal_nodes : array_like
        Array of interior points indices
    weight_function : callable
        Function to calculate weights
    
    Returns:
    --------
    coeffs_for_first_derivative : ndarray
    coeffs_for_second_derivative : ndarray
    """
    num_internal_nodes = len(internal_nodes)
    total_nodes = len(x)
    coeffs_for_first_derivative = []
    coeffs_for_second_derivative = []
    
    for i_node in range(num_internal_nodes):
        C = np.zeros((2, 2))
        cont = 0

        n_neighbors = stars[internal_nodes[i_node]].sum()
        w_ent = np.zeros((2, n_neighbors))
        
        # Implement algorithm
        for node in range(total_nodes):
            if stars[internal_nodes[i_node], node] == 1:
                h = x[node] - x[internal_nodes[i_node]]
                d = weight_function(h)
                ent = np.array([h, h*h/2])
                w_ent[:, cont] = ent * d
                C += np.outer(ent, ent) * d
                cont += 1
        
        # Solve the system using Cholesky
        R = np.linalg.cholesky(C)
        M = np.linalg.solve(R, np.eye(2))
        coeffs_for_derivatives = (M.T @ M) @ w_ent
        coeffs_for_first_derivative.append(coeffs_for_derivatives[0,:])
        coeffs_for_second_derivative.append(coeffs_for_derivatives[1,:])
    
    return coeffs_for_first_derivative, coeffs_for_second_derivative


def plot_stars(x, stars):
    """
    Plot the stars for a 1D mesh and save it with timestamp.
    
    Parameters:
    -----------
    x : array_like
        Points coordinates
    stars : array_like
        Binary matrix where stars[i,j] = 1 if j is in star of i
    """
    plt.figure(figsize=(12, 4))
    num_points = len(x)
    
    # Plot points
    plt.plot(x, np.zeros_like(x), 'ko', label='Points')
    
    # Plot connections for each star
    for i in range(num_points):
        neighbors = np.where(stars[i] == 1)[0]
        for neighbor in neighbors:
            # Draw line from point i to its neighbor
            plt.plot([x[i], x[neighbor]], [0, 0], 'b-', alpha=0.5)
            
        # Highlight central point of current star
        plt.plot(x[i], 0, 'ro', markersize=8)
        plt.text(x[i], 0.1, f'Point {i}')
        
        # Add small vertical offset for visualization
        plt.plot(x[neighbors], np.zeros_like(neighbors), 'go', markersize=6)
    
    plt.grid(True)
    plt.title('Stars Visualization')
    plt.xlabel('x')
    plt.legend()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plot
    plt.savefig(output_dir / f"stars_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
