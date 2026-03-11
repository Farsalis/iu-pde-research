"""Tests for iu_pde_research.core.forcing.

Covers compute_exp_decay_cutoff, compute_F_hat_j,
compute_forcing_contribution_at_time, compute_forcing_truncation_constant_C,
compute_n_k, and make_F_x_from_F.
"""

import math

import numpy as np
import pytest

from iu_pde_research.core.forcing import (
    compute_exp_decay_cutoff,
    compute_F_hat_j,
    compute_forcing_contribution_at_time,
    compute_forcing_truncation_constant_C,
    compute_n_k,
)
from iu_pde_research.core.forcing_autodiff import make_F_x_from_F


def test_compute_F_hat_j_zero_forcing() -> None:
    """compute_F_hat_j returns 0 when F(x,t)=0."""

    def F_zero(x: float, t: float) -> float:
        return 0.0

    result = compute_F_hat_j(F_zero, 1, 0.5)
    assert result == pytest.approx(0.0)


def test_compute_F_hat_j_simple_F_finite() -> None:
    """compute_F_hat_j returns finite value for F(x,t)=sin(x)."""

    def F_sin_x(x: float, t: float) -> float:
        return np.sin(x)

    result = compute_F_hat_j(F_sin_x, 1, 0.0)
    assert np.isfinite(result)
    assert isinstance(result, (float, np.floating))


def test_compute_F_hat_j_sin_x_mode1_positive() -> None:
    """For F(x,t)=sin(x), mode j=1 should give positive coefficient."""

    def F_sin_x(x: float, t: float) -> float:
        return np.sin(x)

    result = compute_F_hat_j(F_sin_x, 1, 0.0)
    assert result > 0


def test_compute_forcing_contribution_at_time_zero_F() -> None:
    """compute_forcing_contribution_at_time returns 0 when F is zero."""

    def F_zero(x: float, t: float) -> float:
        return 0.0

    result = compute_forcing_contribution_at_time(
        F_zero, tk=0.5, x0=np.pi / 2, n_terms=5
    )
    assert result == pytest.approx(0.0)


def test_compute_forcing_contribution_at_time_simple_F_finite() -> None:
    """compute_forcing_contribution_at_time returns finite value for simple F."""

    def F_simple(x: float, t: float) -> float:
        return np.sin(x) * np.exp(-t)

    result = compute_forcing_contribution_at_time(
        F_simple, tk=0.2, x0=np.pi / 2, n_terms=5
    )
    assert np.isfinite(result)


def test_compute_n_k_returns_ceil_integer() -> None:
    """compute_n_k returns integer >= ceil(C * exp((k+1)^2 t_k/2))."""
    C = 1.0
    k = 1
    t_k = 0.5
    raw = C * math.exp((k + 1) ** 2 * t_k / 2.0)
    expected = math.ceil(raw)
    result = compute_n_k(k, t_k, C)
    assert result == expected
    assert isinstance(result, int)


def test_compute_n_k_zero_C_returns_zero() -> None:
    """compute_n_k returns 0 when C <= 0."""
    assert compute_n_k(1, 0.5, 0.0) == 0
    assert compute_n_k(2, 0.3, -1.0) == 0


def test_compute_n_k_negative_t_k_returns_zero() -> None:
    """compute_n_k returns 0 when t_k < 0."""
    assert compute_n_k(1, -0.1, 1.0) == 0


def test_compute_forcing_truncation_constant_C_simple_F_x() -> None:
    """C = (2/pi) * max_s ||F_x(·,s)||_{L^1(0,pi)} for F_x(x,s)=cos(x)."""

    # ||cos(·)||_{L^1(0,pi)} = 2, so C = (2/pi)*2 = 4/pi.
    def F_x_cos(x: float, s: float) -> float:
        return np.cos(x)

    C = compute_forcing_truncation_constant_C(F_x_cos, T=1.0, L=np.pi)
    expected = 4.0 / np.pi
    assert C == pytest.approx(expected, rel=1e-5)


def test_make_F_x_from_F_jax_sin_exp() -> None:
    """make_F_x_from_F yields ∂/∂x(sin(x)*exp(t)) = cos(x)*exp(t)."""
    import jax.numpy as jnp

    def F_jax(x: float, t: float) -> float:
        return jnp.sin(x) * jnp.exp(t)  # JAX ops for grad

    F_x = make_F_x_from_F(F_jax)
    x_val, s_val = 0.5, 0.2
    result = F_x(x_val, s_val)
    expected = math.cos(x_val) * math.exp(s_val)
    assert result == pytest.approx(expected, rel=1e-5)
    assert isinstance(result, float)


# ── compute_exp_decay_cutoff ──────────────────────────────────────────


def test_compute_exp_decay_cutoff_known_value() -> None:
    """For tk=1.0, L=pi, tol=1e-12: N = ceil(sqrt(12*ln(10))) = 6."""
    expected = math.ceil(math.sqrt(12 * math.log(10)))
    assert expected == 6
    result = compute_exp_decay_cutoff(tk=1.0, L=np.pi, tol=1e-12)
    assert result == expected


def test_compute_exp_decay_cutoff_zero_tk_returns_zero() -> None:
    """Cutoff is 0 when tk <= 0 (no time elapsed)."""
    assert compute_exp_decay_cutoff(tk=0.0) == 0
    assert compute_exp_decay_cutoff(tk=-1.0) == 0


def test_compute_exp_decay_cutoff_monotone_in_tk() -> None:
    """Smaller tk requires more full-integral modes (larger N)."""
    n_large_tk = compute_exp_decay_cutoff(tk=1.0)
    n_small_tk = compute_exp_decay_cutoff(tk=0.01)
    assert n_small_tk > n_large_tk


def test_compute_exp_decay_cutoff_general_L() -> None:
    """For L != pi the eigenvalue scaling adjusts N."""
    n_pi = compute_exp_decay_cutoff(tk=0.5, L=np.pi)
    n_2pi = compute_exp_decay_cutoff(tk=0.5, L=2 * np.pi)
    assert n_2pi > n_pi


# ── Two-strategy forcing consistency ─────────────────────────────────


def test_two_strategy_matches_brute_force() -> None:
    """Two-strategy result is close to a brute-force full-integral baseline.

    Uses F(x,t) = x*(pi-x)*exp(-t), a forcing with many nonzero Fourier
    modes, and enough n_terms so that the cutoff N < n_terms exercises both
    strategies.  The reference is computed by summing the full integral for
    every mode up to n_terms (i.e. the old algorithm).
    """

    def F_rich(x: float, t: float) -> float:
        return float(x * (np.pi - x) * np.exp(-t))

    tk = 0.5
    x0 = np.pi / 2
    n_terms = 15

    cutoff = compute_exp_decay_cutoff(tk, np.pi)
    assert cutoff < n_terms, "Test requires N < n_terms to exercise both paths"

    result = compute_forcing_contribution_at_time(F_rich, tk, x0, n_terms)
    assert np.isfinite(result)

    # Brute-force reference: full integral for all j = 1..n_terms
    n_s = 24
    s_grid = np.linspace(0.0, tk, n_s)
    ref = 0.0
    for j in range(1, n_terms + 1):
        fhats = np.array([compute_F_hat_j(F_rich, j, float(s)) for s in s_grid])
        kernel = np.exp(-(j**2) * (tk - s_grid))
        integral_j = float(np.trapezoid(fhats * kernel, s_grid))
        ref += integral_j * np.sin(j * x0)

    assert result == pytest.approx(ref, rel=1e-4)
