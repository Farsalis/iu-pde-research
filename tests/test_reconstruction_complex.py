"""Tests for complex-valued homogeneous modified t_j reconstruction.

References: agent_references/PDE_Meeting_04-08_Writeup_SF.pdf (sine recurrence (3)-(4)
and equivalent exponential form (7)-(8)).
"""

import numpy as np

from iu_pde_research.core.reconstruction import (
    compute_modified_t_j,
    compute_modified_t_j_homogeneous_complex,
    compute_modified_t_j_nonhomogeneous,
)
from iu_pde_research.core.utils import get_default_f_true


def test_homogeneous_complex_shapes_and_finite(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """compute_modified_t_j_homogeneous_complex returns complex arrays matching x_grid."""

    def f_real_as_complex(x: float) -> complex:
        return complex(get_default_f_true(x), 0.0)

    f_true_vals, bar_vals = compute_modified_t_j_homogeneous_complex(
        small_n, T, x0, x_grid, f_true=f_real_as_complex
    )
    assert f_true_vals.shape == x_grid.shape
    assert bar_vals.shape == x_grid.shape
    assert f_true_vals.dtype == np.complex128
    assert bar_vals.dtype == np.complex128
    assert np.all(np.isfinite(f_true_vals.real))
    assert np.all(np.isfinite(f_true_vals.imag))
    assert np.all(np.isfinite(bar_vals.real))
    assert np.all(np.isfinite(bar_vals.imag))


def test_homogeneous_complex_real_f_matches_real_algorithm(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """Real initial data as complex should match compute_modified_t_j on real parts."""

    def f_real_as_complex(x: float) -> complex:
        return complex(get_default_f_true(x), 0.0)

    f_c, bar_c = compute_modified_t_j_homogeneous_complex(
        small_n,
        T,
        x0,
        x_grid,
        f_true=f_real_as_complex,
        L=np.pi,
        n_terms=20,
    )
    f_r, bar_r = compute_modified_t_j(small_n, T, x0, x_grid, f_true=get_default_f_true)
    np.testing.assert_allclose(f_c.real, f_r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(f_c.imag, np.zeros_like(f_r), atol=1e-12)
    np.testing.assert_allclose(bar_c.real, bar_r, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(bar_c.imag, np.zeros_like(bar_r), rtol=0, atol=1e-6)


def test_homogeneous_complex_matches_nonhomogeneous_real_f_general_l(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """Real f with general domain length: complex API matches non-homogeneous F=None."""

    L = 2.0
    x_grid_L = np.linspace(0.01, L - 0.01, 50)
    x0_L = L / 3.0

    def f_real_as_complex(x: float) -> complex:
        return complex(np.sin(np.pi * x / L) + 0.25 * np.sin(3 * np.pi * x / L), 0.0)

    def f_real_nh(x: float) -> float:
        return float(np.sin(np.pi * x / L) + 0.25 * np.sin(3 * np.pi * x / L))

    f_c, bar_c = compute_modified_t_j_homogeneous_complex(
        small_n, T, x0_L, x_grid_L, f_true=f_real_as_complex, L=L
    )
    f_nh, bar_nh = compute_modified_t_j_nonhomogeneous(
        small_n, T, x0_L, x_grid_L, F=None, f_true=f_real_nh, L=L
    )
    np.testing.assert_allclose(f_c.real, f_nh, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bar_c.real, bar_nh, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(bar_c.imag, np.zeros_like(bar_nh), rtol=0, atol=1e-6)


def test_homogeneous_complex_linearity(
    x_grid: np.ndarray, small_n: int, T: float, x0: float
) -> None:
    """If f = alpha * g with alpha in C, bar_f scales by alpha (for fixed truncation)."""

    def g(x: float) -> float:
        return float(np.sin(2 * x) + 0.3 * np.sin(4 * x))

    alpha = 1.0 + 2.0j

    def f_scaled(x: float) -> complex:
        return alpha * g(x)

    _, bar_alpha = compute_modified_t_j_homogeneous_complex(
        small_n, T, x0, x_grid, f_true=f_scaled, L=np.pi
    )

    def g_complex(x: float) -> complex:
        return complex(g(x), 0.0)

    _, bar_g = compute_modified_t_j_homogeneous_complex(
        small_n, T, x0, x_grid, f_true=g_complex, L=np.pi
    )
    np.testing.assert_allclose(bar_alpha, alpha * bar_g, rtol=0, atol=5e-8)


def test_sine_vs_exponential_recurrence_base_and_step() -> None:
    """PDF (3)-(4) vs (7)-(8): same base and recursive formula at a fixed x0."""

    x0 = 0.7
    tk = 0.15
    u_at_t = 0.25 - 0.4j

    sin_x0 = np.sin(x0)
    # (3) base
    bar_b1_sine = np.exp(tk) * u_at_t / sin_x0
    denom_exp = np.exp(1j * x0) - np.exp(-1j * x0)
    bar_b1_exp = 2j * np.exp(tk) / denom_exp * u_at_t
    np.testing.assert_allclose(bar_b1_sine, bar_b1_exp, rtol=0, atol=1e-12)

    bar_b1 = 0.5 + 0.25j
    # (4) k=2
    j = 1
    inner_sine = u_at_t - np.exp(-(j**2) * tk) * bar_b1 * np.sin(j * x0)
    bar_b2_sine = np.exp(4 * tk) / np.sin(2 * x0) * inner_sine

    inner_exp = u_at_t - np.exp(-(j**2) * tk) * bar_b1 * (-0.5j) * (
        np.exp(1j * j * x0) - np.exp(-1j * j * x0)
    )
    bar_b2_exp = 2j * np.exp(4 * tk) / (np.exp(2j * x0) - np.exp(-2j * x0)) * inner_exp
    np.testing.assert_allclose(bar_b2_sine, bar_b2_exp, rtol=0, atol=1e-12)
