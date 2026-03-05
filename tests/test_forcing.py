"""Tests for iu_pde_research.core.forcing.

Covers compute_F_hat_j and compute_forcing_contribution_at_time for zero and
simple forcing functions.
"""

import numpy as np
import pytest

from iu_pde_research.core.forcing import (
    compute_F_hat_j,
    compute_forcing_contribution_at_time,
)


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
