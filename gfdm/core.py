import numpy as np

from .utils import build_matrices, create_stars, plot_stars
from .weights import weight_function


class GFDMSolver:
    """Main solver class for Generalized Finite Difference Method"""

    def __init__(self, num_neighbors=2):
        self.num_neighbors = num_neighbors

    def build_matrices(self, x, stars, weight_fn=None):
        """Build matrices for GFDM computation"""
        if weight_fn is None:
            weight_fn = weight_function

        return build_matrices(x, stars, weight_fn)

    def plot_stars(self, x, stars):
        "Plot stars and save graph"
        plot_stars(x, stars)

    def create_stars(self, x):
        """Create stars for each point"""
        return create_stars(x, self.num_neighbors)
