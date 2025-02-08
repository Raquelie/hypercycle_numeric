from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_3d(num_steps, delta_t, x, sol):
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
    - Saves the plot as 'inf_model_YYYYMMDD_HHMMSS.png'
    """

    print("\nCreating 3D visualization...")
    T = delta_t * num_steps
    X, T = np.meshgrid(x, np.arange(0, T, delta_t))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, T, sol, cmap="viridis", antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    ax.set_xlim(np.max(X), 0)  # Reverse x axis direction
    ax.view_init(elev=20, azim=-45)
    ax.grid(True, alpha=0.3)

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save plot
    output_file = output_dir / f"inf_model_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n Plot saved to {output_file}")
