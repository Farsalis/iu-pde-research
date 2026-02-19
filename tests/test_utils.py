"""Tests for iu_pde_research.core.utils."""

import numpy as np
import pytest

from iu_pde_research.core.utils import (
    compute_t_vec,
    get_default_f_true,
    get_f_hat_true,
)


def test_get_default_f_true_scalar_in_scalar_out() -> None:
    """get_default_f_true returns a float for scalar input."""
    x = np.pi / 2
    result = get_default_f_true(x)
    assert isinstance(result, (float, np.floating))
    assert np.isfinite(result)


def test_get_default_f_true_known_value() -> None:
    """get_default_f_true at x=pi/2 is sin(2*pi/2)+0.5*sin(5*pi/2) = 0 + 0.5 = 0.5."""
    x = np.pi / 2
    result = get_default_f_true(x)
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize("j", [1, 2, 3])
def test_get_f_hat_true_finite_and_reasonable(j: int) -> None:
    """get_f_hat_true returns finite values for modes 1,2,3 with default f_true."""
    result = get_f_hat_true(j, get_default_f_true)
    assert np.isfinite(result)
    assert isinstance(result, (float, np.floating))


def test_get_f_hat_true_sin2x_j2_positive() -> None:
    """For f(x)=sin(2x), Fourier coefficient j=2 should be positive (dominant mode)."""

    def f_sin2(x: float) -> float:
        return np.sin(2 * x)

    result = get_f_hat_true(2, f_sin2)
    assert result > 0
    assert np.isfinite(result)


@pytest.mark.parametrize("n,T", [(2, 1.0), (4, 2.0), (5, 1.0)])
def test_compute_t_vec_shape_and_dtype(n: int, T: float) -> None:
    """compute_t_vec returns shape (n,) and float dtype."""
    t_vec = compute_t_vec(n, T)
    assert t_vec.shape == (n,)
    assert t_vec.dtype in (np.float64, np.float32)


@pytest.mark.parametrize("n,T", [(3, 1.0), (4, 2.0)])
def test_compute_t_vec_positive_and_increasing(n: int, T: float) -> None:
    """compute_t_vec returns strictly positive, strictly increasing values."""
    t_vec = compute_t_vec(n, T)
    assert np.all(t_vec > 0)
    assert np.all(np.diff(t_vec) > 0)
