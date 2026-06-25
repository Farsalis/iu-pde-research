"""Tests for iu_pde_research.visualization.plots.

Covers plot_reconstruction (save/show) and plot_reconstruction_with_errors
(returned metrics and zero-error case).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from iu_pde_research.visualization.plots import (
    plot_reconstruction,
    plot_reconstruction_with_errors,
)


def test_plot_reconstruction_save_no_show(x_grid: np.ndarray, tmp_path: Path) -> None:
    """plot_reconstruction with show=False and save_path creates a non-empty file."""
    f_true_vals = np.sin(2 * x_grid) + 0.5 * np.sin(5 * x_grid)
    bar_f_n_vals = f_true_vals + 0.01 * np.random.default_rng(42).random(x_grid.size)
    save_path = tmp_path / "recon.png"
    plot_reconstruction(
        x_grid, f_true_vals, bar_f_n_vals, save_path=str(save_path), show=False
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    plt.close("all")


def test_plot_reconstruction_with_errors_returns_tuple(
    x_grid: np.ndarray, tmp_path: Path
) -> None:
    """plot_reconstruction_with_errors returns (l2, linf, mse) with non-negative values."""
    f_true_vals = np.sin(2 * x_grid)
    bar_f_n_vals = f_true_vals + 0.1 * np.ones_like(x_grid)
    save_path = tmp_path / "recon_err.png"
    l2, linf, mse = plot_reconstruction_with_errors(
        x_grid, f_true_vals, bar_f_n_vals, save_path=str(save_path), show=False
    )
    assert isinstance(l2, (float, np.floating))
    assert isinstance(linf, (float, np.floating))
    assert isinstance(mse, (float, np.floating))
    assert l2 >= 0
    assert linf >= 0
    assert mse >= 0
    plt.close("all")


def test_plot_reconstruction_with_errors_zero_error_metrics(
    x_grid: np.ndarray,
) -> None:
    """When bar_f_n_vals equals f_true_vals, L2 and MSE are zero (linf may be 0)."""
    f_true_vals = np.sin(2 * x_grid)
    l2, linf, mse = plot_reconstruction_with_errors(
        x_grid, f_true_vals, f_true_vals.copy(), show=False
    )
    assert l2 == pytest.approx(0.0)
    assert linf == pytest.approx(0.0)
    assert mse == pytest.approx(0.0)
    plt.close("all")
