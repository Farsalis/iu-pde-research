"""Tests for JAX-accelerated reconstruction routines.

Covers basic shape checks and numerical agreement with the CPU implementation
for the homogeneous modified t_j method.
"""

import numpy as np
import pytest

from iu_pde_research.core import (
    compute_modified_t_j,
    compute_modified_t_j_jax,
)


@pytest.mark.parametrize("n", [4, 8])
def test_compute_modified_t_j_jax_shapes_and_finite(
    x_grid: np.ndarray,
    n: int,
    T: float,
    x0: float,
) -> None:
    """JAX version returns finite arrays matching x_grid.shape."""
    f_true_vals_jax, bar_f_n_vals_jax = compute_modified_t_j_jax(n, T, x0, x_grid)
    assert f_true_vals_jax.shape == x_grid.shape
    assert bar_f_n_vals_jax.shape == x_grid.shape
    assert np.all(np.isfinite(f_true_vals_jax))
    assert np.all(np.isfinite(bar_f_n_vals_jax))


@pytest.mark.parametrize("n", [4, 8])
def test_compute_modified_t_j_jax_matches_cpu(
    x_grid: np.ndarray,
    n: int,
    T: float,
    x0: float,
) -> None:
    """JAX and CPU implementations agree for small n on default f_true."""
    f_true_cpu, bar_cpu = compute_modified_t_j(n, T, x0, x_grid)
    f_true_jax, bar_jax = compute_modified_t_j_jax(n, T, x0, x_grid)

    np.testing.assert_allclose(f_true_cpu, f_true_jax, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(bar_cpu, bar_jax, rtol=1e-5, atol=1e-6)
