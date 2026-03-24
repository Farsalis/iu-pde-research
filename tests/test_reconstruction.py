"""Tests for iu_pde_research.core.reconstruction.

Covers shapes, finiteness, parametrized n, non-homogeneous vs homogeneous
agreement, and linear t_j homogeneous reconstruction.
"""

import numpy as np
import pytest

from iu_pde_research.core.reconstruction import (
    compute_linear_t_j_homogeneous,
    compute_modified_t_j,
    compute_modified_t_j_nonhomogeneous,
)


def test_compute_modified_t_j_shapes(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """compute_modified_t_j returns arrays matching x_grid.shape."""
    f_true_vals, bar_f_n_vals = compute_modified_t_j(small_n, T, x0, x_grid)
    assert f_true_vals.shape == x_grid.shape
    assert bar_f_n_vals.shape == x_grid.shape


def test_compute_modified_t_j_finite(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """compute_modified_t_j returns finite values."""
    f_true_vals, bar_f_n_vals = compute_modified_t_j(small_n, T, x0, x_grid)
    assert np.all(np.isfinite(f_true_vals))
    assert np.all(np.isfinite(bar_f_n_vals))


@pytest.mark.parametrize("n", [2, 4])
def test_compute_modified_t_j_parametrized_n(
    x_grid: np.ndarray, n: int, T: float, x0: float
) -> None:
    """compute_modified_t_j runs for small n=2 and n=4."""
    f_true_vals, bar_f_n_vals = compute_modified_t_j(n, T, x0, x_grid)
    assert f_true_vals.shape == x_grid.shape
    assert bar_f_n_vals.shape == x_grid.shape


def test_compute_modified_t_j_reconstruction_improves_with_n(
    x_grid: np.ndarray, T: float, x0: float
) -> None:
    """Reconstruction error (L2) decreases as n increases."""
    _, bar4 = compute_modified_t_j(4, T, x0, x_grid)
    _, bar8 = compute_modified_t_j(8, T, x0, x_grid)
    f_true = np.sin(2 * x_grid) + 0.5 * np.sin(5 * x_grid)
    err4 = np.sqrt(np.trapezoid((bar4 - f_true) ** 2, x_grid))
    err8 = np.sqrt(np.trapezoid((bar8 - f_true) ** 2, x_grid))
    assert err8 < err4


def test_compute_modified_t_j_nonhomogeneous_F_none_matches_homogeneous(
    x_grid: np.ndarray, small_n: int, T: float, x0: float, f_true_simple
) -> None:
    """With F=None, nonhomogeneous should match homogeneous (same shapes and close values)."""
    f_h, bar_h = compute_modified_t_j(small_n, T, x0, x_grid, f_true=f_true_simple)
    f_nh, bar_nh = compute_modified_t_j_nonhomogeneous(
        small_n, T, x0, x_grid, F=None, f_true=f_true_simple
    )
    assert f_nh.shape == x_grid.shape
    assert bar_nh.shape == x_grid.shape
    np.testing.assert_allclose(f_h, f_nh)
    np.testing.assert_allclose(bar_h, bar_nh)


def test_compute_modified_t_j_nonhomogeneous_with_F_finite(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """Nonhomogeneous with simple F and F_x returns finite values."""

    def F_simple(x: float, t: float) -> float:
        return np.sin(x) * np.exp(-t)

    def F_x_simple(x: float, s: float) -> float:
        return np.cos(x) * np.exp(-s)

    f_vals, bar_vals = compute_modified_t_j_nonhomogeneous(
        small_n, T, x0, x_grid, F=F_simple, F_x=F_x_simple
    )
    assert np.all(np.isfinite(f_vals))
    assert np.all(np.isfinite(bar_vals))
    assert f_vals.shape == x_grid.shape
    assert bar_vals.shape == x_grid.shape


def test_compute_linear_t_j_homogeneous_shapes(
    x_grid: np.ndarray, small_n: int, x0: float
) -> None:
    """compute_linear_t_j_homogeneous returns arrays matching x_grid.shape."""
    t0 = 1e-3
    f_true_vals, bar_f_n_vals = compute_linear_t_j_homogeneous(small_n, t0, x0, x_grid)
    assert f_true_vals.shape == x_grid.shape
    assert bar_f_n_vals.shape == x_grid.shape


@pytest.mark.parametrize("n,t0", [(4, 1e-3), (6, 0.01)])
def test_compute_linear_t_j_homogeneous_parametrized(
    x_grid: np.ndarray, n: int, t0: float, x0: float
) -> None:
    """compute_linear_t_j_homogeneous runs for small n and t0."""
    f_true_vals, bar_f_n_vals = compute_linear_t_j_homogeneous(n, t0, x0, x_grid)
    assert np.all(np.isfinite(f_true_vals))
    assert np.all(np.isfinite(bar_f_n_vals))
