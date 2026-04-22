"""Tests for CuPy-accelerated reconstruction routines.

Covers basic shape checks and numerical agreement with the CPU
implementation for the non-homogeneous modified t_j method.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from iu_pde_research.core import (
    compute_modified_t_j_nonhomogeneous,
    compute_modified_t_j_nonhomogeneous_cupy,
)


@pytest.mark.parametrize("n", [4, 8])
def test_compute_modified_t_j_nonhomogeneous_cupy_shapes_and_finite(
    x_grid: np.ndarray,
    n: int,
    T: float,
    x0: float,
) -> None:
    """CuPy version returns finite arrays matching x_grid.shape."""
    f_true_vals, bar_f_n_vals = compute_modified_t_j_nonhomogeneous_cupy(
        n, T, x0, x_grid
    )
    assert f_true_vals.shape == x_grid.shape
    assert bar_f_n_vals.shape == x_grid.shape
    assert np.all(np.isfinite(f_true_vals))
    assert np.all(np.isfinite(bar_f_n_vals))


@pytest.mark.parametrize("n", [4, 8])
def test_compute_modified_t_j_nonhomogeneous_cupy_matches_cpu(
    x_grid: np.ndarray,
    n: int,
    T: float,
    x0: float,
) -> None:
    """CuPy and CPU implementations agree for small n on default f_true."""
    f_true_cpu, bar_cpu = compute_modified_t_j_nonhomogeneous(n, T, x0, x_grid)
    f_true_cupy, bar_cupy = compute_modified_t_j_nonhomogeneous_cupy(n, T, x0, x_grid)

    np.testing.assert_allclose(f_true_cpu, f_true_cupy, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(bar_cpu, bar_cupy, rtol=1e-5, atol=1e-6)
