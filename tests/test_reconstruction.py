"""Tests for iu_pde_research.core.reconstruction.

Covers shapes, finiteness, parametrized n, non-homogeneous vs homogeneous
agreement, and linear t_j homogeneous reconstruction.
"""

# mypy: disable-error-code="misc"

from collections.abc import Callable

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


def test_compute_modified_t_j_reconstruction_l2_error_finite(
    x_grid: np.ndarray, T: float
) -> None:
    """Finite L2 error vs default spectrum for moderate n (sensor away from sin zeros)."""
    x0_safe = 1.1
    f_true = np.sin(2 * x_grid) + 0.5 * np.sin(5 * x_grid)
    for n in (4, 8):
        _, bar = compute_modified_t_j(n, T, x0_safe, x_grid)
        err = np.sqrt(np.trapezoid((bar - f_true) ** 2, x_grid))
        assert np.isfinite(err)
        assert err < 15.0


def test_compute_modified_t_j_nonhomogeneous_F_none_matches_homogeneous(
    x_grid: np.ndarray,
    small_n: int,
    T: float,
    x0: float,
    f_true_simple: Callable[[float], float],
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
    """Nonhomogeneous with simple F returns finite values."""

    def F_simple(x: float, t: float) -> float:
        return float(np.sin(x) * np.exp(-t))

    f_vals, bar_vals = compute_modified_t_j_nonhomogeneous(
        small_n, T, x0, x_grid, F=F_simple
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
