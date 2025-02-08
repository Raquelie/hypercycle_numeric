from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_3d(num_steps, delta_t, x, sol, model):
    """
    Create and save a 3D surface plot of the solution to the hypercycle models.

    The resulting plot is automatically saved with a timestamp in the output directory.

    Parameters
    ----------
    num_steps : int
        Number of time steps in the simulation
    delta_t : float
        Time step size (dt)
    x : np.ndarray
        1D array of spatial points
    sol : np.ndarray
        2D array of shape (num_steps, len(x)) containing the solution values
        where rows represent time steps and columns represent spatial points

    Returns
    -------
    None
        The function saves the plot to a PNG file but doesn't return any value

    Output
    ------
    - Creates a directory 'output' if it doesn't exist
    - Saves the plot as '{model}_YYYYMMDD_HHMMSS.png'
    """
    print("\nCreating 3D visualization...")
    T = delta_t * num_steps
    X, T = np.meshgrid(x, np.arange(0, T, delta_t))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if model == "inf_model":
        ax.plot_surface(X, T, sol, cmap="viridis", antialiased=True)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u")
        ax.set_xlim(np.max(X), 0)  # Reverse x axis direction
        ax.view_init(elev=20, azim=-45)
        ax.grid(True, alpha=0.3)
    else:
        ax.plot_surface(T, X, sol, cmap="viridis", antialiased=True)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_zlabel("v")
        ax.set_xlim(np.max(T), 0)  # Reverse t axis direction
        ax.set_ylim(np.max(X), 0)  # Reverse x axis direction

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save plot
    output_file = output_dir / f"{model}_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n Plot saved to {output_file}")


def plot_multiple(num_steps, delta_t, x, n_molecules, sol):
    """
    Create and save a 3D surface plot of the solution to the hypercycle models.

    The resulting plot is automatically saved with a timestamp in the output directory.

    Parameters
    ----------
    num_steps : int
        Number of time steps in the simulation
    delta_t : float
        Time step size (dt)
    x : np.ndarray
        1D array of spatial points
    n_molecules: int
        Number of molecules in the simulation
    sol : np.ndarray
        2D array of shape (num_steps, len(x)) containing the solution values
        where rows represent time steps and columns represent spatial points

    Returns
    -------
    None
        The function saves the plot to a PNG file but doesn't return any value

    Output
    ------
    - Creates a directory 'output' if it doesn't exist
    - Saves the plot as 'n_mol_YYYYMMDD_HHMMSS.png'
    """
    print("\nCreating 3D visualization...")
    T = delta_t * num_steps
    X, T = np.meshgrid(x, np.arange(0, T, delta_t))

    # Calculate number of rows and columns for subplots
    n_cols = min(2, n_molecules)  # Maximum 2 columns
    n_rows = (n_molecules + n_cols - 1) // n_cols

    # Create figure with 3D subplots
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # Create subplots for each molecule
    for i in range(n_molecules):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")

        # Plot surface for current molecule
        ax.plot_surface(T, X, sol[i, :, :], cmap="viridis", antialiased=True)

        ax.set_xlim(np.max(T), 0)  # Reverse t axis direction
        # ax.set_ylim(np.max(X), 0)  # Reverse x axis direction
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_zlabel("v")
        ax.set_title(f"Molecule {i+1}")

    plt.tight_layout(pad=3.0)

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save plot
    output_file = output_dir / f"n_mol_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n Plot saved to {output_file}")
