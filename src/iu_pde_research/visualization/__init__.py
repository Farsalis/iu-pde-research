"""Visualization modules for PDE reconstruction results.

Plotting and interactive visualization: compare true vs reconstructed initial
condition, report error metrics (L2, Linf, MSE), and adjust parameters n, T, x0
in real time via text boxes.
"""

from iu_pde_research.visualization.plots import (
    create_interactive_plot,
    plot_reconstruction,
    plot_reconstruction_with_errors,
)

# Export functions available in this package
__all__ = [
    "create_interactive_plot",
    "plot_reconstruction",
    "plot_reconstruction_with_errors",
]
