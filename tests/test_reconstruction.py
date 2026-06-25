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
from iu_pde_research.core.utils import compute_t_vec, get_f_hat_true


def _neumann_homogeneous_sensor(
    f_true: Callable[[float], float],
    x0: float,
    t: float,
    *,
    L: float = np.pi,
    n_terms: int = 20,
) -> float:
    """Synthetic Neumann sensor reading u(x0, t) for a truncated cosine series."""
    reading = get_f_hat_true(0, f_true, L, bc="neumann")
    for j in range(1, n_terms + 1):
        f_hat_j = get_f_hat_true(j, f_true, L, bc="neumann")
        reading += f_hat_j * np.exp(-(j**2) * t) * np.cos(j * x0)
    return float(reading)


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
    """Nonhomogeneous with simple F and F_x returns finite values."""

    def F_simple(x: float, t: float) -> float:
        return float(np.sin(x) * np.exp(-t))

    def F_x_simple(x: float, s: float) -> float:
        return np.cos(x) * np.exp(-s)

    f_vals, bar_vals = compute_modified_t_j_nonhomogeneous(
        small_n, T, x0, x_grid, F=F_simple, F_x=F_x_simple
    )
    assert np.all(np.isfinite(f_vals))
    assert np.all(np.isfinite(bar_vals))
    assert f_vals.shape == x_grid.shape
    assert bar_vals.shape == x_grid.shape


def test_compute_modified_t_j_neumann_recovers_cosine_series() -> None:
    """Neumann homogeneous reconstruction keeps the mean and cosine modes."""
    x_grid = np.linspace(0.01, np.pi - 0.01, 80)
    x0_safe = 0.7

    def f_neumann(x: float) -> float:
        return float(1.25 + 0.5 * np.cos(x))

    f_vals, bar_vals = compute_modified_t_j(
        n=4,
        T=0.1,
        x0=x0_safe,
        x_grid=x_grid,
        f_true=f_neumann,
        bc="neumann",
    )

    np.testing.assert_allclose(f_vals, bar_vals, atol=1e-7, rtol=1e-7)


def test_compute_modified_t_j_neumann_accepts_known_zero_mode() -> None:
    """Supplying f_hat_0 overrides the mean used in the Neumann reconstruction."""
    x_grid = np.linspace(0.01, np.pi - 0.01, 80)
    x0_safe = 0.7

    def f_neumann(x: float) -> float:
        return float(1.25 + 0.5 * np.cos(x))

    _, bar_vals = compute_modified_t_j(
        n=4,
        T=0.1,
        x0=x0_safe,
        x_grid=x_grid,
        f_true=f_neumann,
        bc="neumann",
        f_hat_0=1.25,
    )

    expected = np.asarray([f_neumann(float(x)) for x in x_grid], dtype=np.float64)
    np.testing.assert_allclose(expected, bar_vals, atol=1e-7, rtol=1e-7)


def test_compute_modified_t_j_nonhomogeneous_neumann_constant_forcing() -> None:
    """Constant forcing is removed through G_0 before Neumann inversion."""
    x_grid = np.linspace(0.01, np.pi - 0.01, 80)
    x0_safe = 0.7

    def f_neumann(x: float) -> float:
        return float(1.25 + 0.5 * np.cos(x))

    def F_constant(x: float, t: float) -> float:
        return 2.0

    def F_x_zero(x: float, s: float) -> float:
        return 0.0

    f_vals, bar_vals = compute_modified_t_j_nonhomogeneous(
        n=4,
        T=0.1,
        x0=x0_safe,
        x_grid=x_grid,
        F=F_constant,
        F_x=F_x_zero,
        f_true=f_neumann,
        bc="neumann",
    )

    np.testing.assert_allclose(f_vals, bar_vals, atol=1e-7, rtol=1e-7)


def test_compute_modified_t_j_neumann_measurement_zero_mode_homogeneous() -> None:
    """Measurement zero-mode path recovers f for large T and mean equals U_1."""
    x_grid = np.linspace(0.01, np.pi - 0.01, 80)
    x0_safe = 0.7
    T = 20.0
    n = 4

    def f_neumann(x: float) -> float:
        return float(1.25 + 0.5 * np.cos(x))

    f_vals, bar_vals = compute_modified_t_j(
        n=n,
        T=T,
        x0=x0_safe,
        x_grid=x_grid,
        f_true=f_neumann,
        bc="neumann",
        zero_mode="measurement",
    )

    expected = np.asarray([f_neumann(float(x)) for x in x_grid], dtype=np.float64)
    np.testing.assert_allclose(bar_vals, expected, atol=0.05, rtol=0.05)

    t_vec = compute_t_vec(n, T)
    u_at_t1 = _neumann_homogeneous_sensor(f_neumann, x0_safe, float(t_vec[0]))
    _, bar_at_mode0 = compute_modified_t_j(
        n=n,
        T=T,
        x0=x0_safe,
        x_grid=np.array([np.pi / 2]),
        f_true=f_neumann,
        bc="neumann",
        zero_mode="measurement",
    )
    np.testing.assert_allclose(float(bar_at_mode0[0]), u_at_t1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(f_vals, expected)


def test_compute_modified_t_j_nonhomogeneous_neumann_measurement_constant_forcing() -> (
    None
):
    """Constant forcing: recovered mean equals U_1 - G_0(t_1) for large T."""
    x_grid = np.linspace(0.01, np.pi - 0.01, 80)
    x0_safe = 0.7
    T = 20.0
    n = 4

    def f_neumann(x: float) -> float:
        return float(1.25 + 0.5 * np.cos(x))

    def F_constant(x: float, t: float) -> float:
        return 2.0

    def F_x_zero(x: float, s: float) -> float:
        return 0.0

    f_vals, bar_vals = compute_modified_t_j_nonhomogeneous(
        n=n,
        T=T,
        x0=x0_safe,
        x_grid=x_grid,
        F=F_constant,
        F_x=F_x_zero,
        f_true=f_neumann,
        bc="neumann",
        zero_mode="measurement",
    )

    expected = np.asarray([f_neumann(float(x)) for x in x_grid], dtype=np.float64)
    np.testing.assert_allclose(bar_vals, expected, atol=0.05, rtol=0.05)

    t_vec = compute_t_vec(n, T)
    t1 = float(t_vec[0])
    u_at_t1 = _neumann_homogeneous_sensor(f_neumann, x0_safe, t1) + 2.0 * t1
    _, bar_at_mode0 = compute_modified_t_j_nonhomogeneous(
        n=n,
        T=T,
        x0=x0_safe,
        x_grid=np.array([np.pi / 2]),
        F=F_constant,
        F_x=F_x_zero,
        f_true=f_neumann,
        bc="neumann",
        zero_mode="measurement",
    )
    np.testing.assert_allclose(
        float(bar_at_mode0[0]), u_at_t1 - 2.0 * t1, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(f_vals, expected)


def test_zero_mode_measurement_requires_neumann(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """zero_mode='measurement' is rejected for Dirichlet bc."""
    with pytest.raises(
        ValueError, match="zero_mode='measurement' requires bc='neumann'"
    ):
        compute_modified_t_j(
            small_n,
            T,
            x0,
            x_grid,
            bc="dirichlet",
            zero_mode="measurement",
        )


def test_zero_mode_measurement_rejects_supplied_f_hat_0() -> None:
    """zero_mode='measurement' cannot be combined with f_hat_0."""
    x_grid = np.linspace(0.01, np.pi - 0.01, 80)
    x0_safe = 0.7

    def f_neumann(x: float) -> float:
        return float(1.25 + 0.5 * np.cos(x))

    with pytest.raises(
        ValueError,
        match="zero_mode='measurement' is incompatible with a supplied f_hat_0",
    ):
        compute_modified_t_j(
            n=4,
            T=0.1,
            x0=x0_safe,
            x_grid=x_grid,
            f_true=f_neumann,
            bc="neumann",
            f_hat_0=1.25,
            zero_mode="measurement",
        )


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
