"""
Main entry point for PDE initial temperature profile reconstruction.

This module provides command-line interface for running various reconstruction
methods.
"""

import sys
import numpy as np

from core.reconstruction import (
    compute_modified_t_j,
    compute_modified_t_j_nonhomogeneous,
    compute_linear_t_j_homogeneous,
)
from visualization.plots import (
    plot_reconstruction,
    plot_reconstruction_with_errors,
    create_interactive_plot,
)


def modified_t_j() -> None:
    """
    Run factorial t_j method for homogeneous case.

    Demonstrates the reconstruction of initial condition using the modified
    factorial t_j  for the homogeneous heat equation.
    """
    n = 20
    T = 5
    x0 = np.sqrt(2)
    x_grid = np.linspace(0, np.pi, 200)
    f_true_vals, bar_f_n_vals = compute_modified_t_j(n, T, x0, x_grid)
    plot_reconstruction(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title="Initial Condition Reconstruction (Modified t_j)",
        save_path="modified_t_j.png",
    )


def modified_t_j_nonhomogeneous() -> None:
    """
    Run factorial t_j method for non-homogeneous case.

    Demonstrates the reconstruction of initial condition using the modified
    factorial t_j method for the non-homogeneous heat equation with a forcing term.
    """
    n = 30
    T = 5.0
    x0 = np.sqrt(2)
    x_grid = np.linspace(0, np.pi, 200)

    def f_true(x):
        """Initial condition function."""
        return np.sin(2 * x) + 0.5 * np.sin(5 * x)

    def F(x, t):
        """Forcing term function."""
        return np.sin(x) * np.exp(-t)

    print("Computing non-homogeneous reconstruction...")
    print(f"Parameters: n={n}, T={T}, x0={x0:.4f}")
    print(f"Forcing term: F(x,t) = sin(x) * exp(-t)")

    f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous(
        n, T, x0, x_grid, F=F, f_true=f_true
    )

    plot_reconstruction_with_errors(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title=(
            "Initial Temperature Profile Reconstruction for Non-Homogeneous Case \n"
            r"$F(x,t)=\sin(x)\exp(-t)$"
        ),
        save_path="modified_t_j_nonhomogeneous.png",
    )


def linear_sequence() -> None:
    """
    Run linear sequence method for reconstruction.

    Demonstrates the reconstruction of initial condition using a linear
    time sequence method.
    """
    n = 20
    t0 = 1e-3  # Tolerance
    x0 = np.sqrt(2)
    x_grid = np.linspace(0, np.pi, 200)

    def f_true(x):
        """Initial condition function."""
        return np.sin(2 * x) + 0.5 * np.sin(5 * x)

    f_true_vals, bar_f_n_vals = compute_linear_t_j_homogeneous(
        n, t0, x0, x_grid, f_true=f_true
    )

    plot_reconstruction(
        x_grid,
        f_true_vals,
        bar_f_n_vals,
        title="Initial Condition Reconstruction (Linear t_j)",
        save_path="linear_sequence.png",
    )


def main() -> None:
    """Main entry point for command-line interface."""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("Starting interactive mode...")
        print("Use the text boxes to adjust n, T, and x0 parameters.")
        create_interactive_plot()
    elif len(sys.argv) > 1 and sys.argv[1] == "--nonhomogeneous":
        print("Running non-homogeneous case...")
        modified_t_j_nonhomogeneous()
    else:
        print("Running Modified t_j method (homogeneous)...")
        modified_t_j()
        print("\nRunning Linear sequence method...")
        linear_sequence()
        print("\nDone!")
        print("\nTo run interactive mode, use: python nhh_re.py --interactive")
        print(
            "To run non-homogeneous case, use: "
            "python nhh_re.py --nonhomogeneous"
        )


if __name__ == "__main__":
    main()
