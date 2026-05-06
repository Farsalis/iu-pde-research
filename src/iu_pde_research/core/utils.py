"""Utility functions and constants for PDE reconstruction.

Provides the default initial condition and Fourier-coefficient helpers
(get_default_f_true, get_f_hat_true with caching), and compute_t_vec for the
modified factorial t_j time sequence. Uses optional numba JIT when available for
faster time-vector computation.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import quad

# Allows us to compile factorials just in time.
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Dummy decorator when numba is not available."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    prange = range

# Cache for Fourier coefficients
_F_HAT_CACHE = {}


def get_default_f_true(x: float | np.ndarray) -> float | np.ndarray:
    """
    Default initial condition function for the homogeneous heat equation.

    Args:
        x: Spatial coordinate (scalar or array).

    Returns:
        Value of the default initial condition at x.
    """
    values = (1/8)*np.sin(2*x)+(1/27)*np.sin(3*x)+(1/64)*np.sin(4 * x)+(1/125)*np.sin(5*x)+(1/216)*np.sin(6*x)
    if np.ndim(values) == 0:
        return float(values)
    return values


def _f_true_sine_series_n(x: float | np.ndarray, n: int) -> float | np.ndarray:
    """Initial condition f(x) = sum_{j=1}^{n} a_j*sin(j*x) with a_1=0 and a_j=1/(2*j^2) for j>=2."""
    out_orig_ic = np.zeros_like(x, dtype=np.float64) if np.ndim(x) != 0 else 0.0
    for j in range(1, int(n) + 1):
        coeff = 0.0 if j == 1 else 1.0 / (2.0 * j * j)
        if np.ndim(x) == 0:
            out_orig_ic = out_orig_ic + coeff * np.sin(
                float(j) * float(x)
            )  # Append shorthand
        else:
            out_orig_ic = out_orig_ic + coeff * np.sin(j * x)
    return out_orig_ic


def get_f_true_sine_series_5(x: float | np.ndarray) -> float | np.ndarray:
    """
    Initial condition f(x) = sum_{j=1}^{5} a_j*sin(j*x) with f_hat_1=0 and a_j=1/(2*j^2) for j>=2.
    """
    return _f_true_sine_series_n(x, 5)


def get_f_true_sine_series_10(x: float | np.ndarray) -> float | np.ndarray:
    """
    Initial condition f(x) = sum_{j=1}^{10} a_j*sin(j*x) with f_hat_1=0 and a_j=1/(2*j^2) for j>=2.
    """
    return _f_true_sine_series_n(x, 10)


def get_f_true_sine_series_15(x: float | np.ndarray) -> float | np.ndarray:
    """
    Initial condition f(x) = sum_{j=1}^{15} a_j*sin(j*x) with f_hat_1=0 and a_j=1/(2*j^2) for j>=2.
    """
    return _f_true_sine_series_n(x, 15)


def _f_true_sine_series_n_cubed(x: float | np.ndarray, n: int) -> float | np.ndarray:
    out_new_ic = np.zeros_like(x, dtype=np.float64) if np.ndim(x) != 0 else 0.0
    for j in range(1, int(n) + 1):
        coeff = 0.0 if j == 1 else 1.0 / (j * j * j)
        if np.ndim(x) == 0:
            # Instance without initial out_new_ic, so specify dtype as float
            out_new_ic = out_new_ic + coeff * np.sin(float(j) * float(x))
        else:
            out_new_ic = out_new_ic + coeff * np.sin(j * x)
    return out_new_ic


def get_f_true_new_sine_series_5(x: float | np.ndarray) -> float | np.ndarray:
    """
    Initial condition f(x) = sum_{j=1}^{15} a_j*sin(j*x) with f_hat_1=0 and a_j=1/(2*j^2) for j>=2.
    """
    return _f_true_sine_series_n_cubed(x, 5)


def get_f_true_new_sine_series_10(x: float | np.ndarray) -> float | np.ndarray:
    """
    Initial condition f(x) = sum_{j=1}^{15} a_j*sin(j*x) with f_hat_1=0 and a_j=1/(2*j^2) for j>=2.
    """
    return _f_true_sine_series_n_cubed(x, 10)


def get_f_true_new_sine_series_15(x: float | np.ndarray) -> float | np.ndarray:
    """
    Initial condition f(x) = sum_{j=1}^{15} a_j*sin(j*x) with f_hat_1=0 and a_j=1/(2*j^2) for j>=2.
    """
    return _f_true_sine_series_n_cubed(x, 15)


# Named initial condition options for the patterns (1/(2*j^2))*sin(j*x) and (1/(j^3))*sin(j*x) .
INITIAL_CONDITION_OPTIONS: dict[
    str, Callable[[float | np.ndarray], float | np.ndarray]
] = {
    "default": get_default_f_true,
    "sine_series_5": get_f_true_sine_series_5,
    "sine_series_10": get_f_true_sine_series_10,
    "sine_series_15": get_f_true_sine_series_15,
    "sine_series_new_5": get_f_true_new_sine_series_5,
    "sine_series_new_10": get_f_true_new_sine_series_10,
    "sine_series_new_15": get_f_true_new_sine_series_15,
}


def get_f_hat_true(
    j: int,
    f_true: Callable[[float], float] | None = None,
    L: float = np.pi,
) -> float:
    """
    Compute Fourier coefficient for dummy mode j.

    Args:
        j: Mode number (positive integer).
        f_true: Initial condition function. If None, uses the default initial
            condition.
        L: Domain length. Defaults to pi.

    Returns:
        Fourier coefficient f_hat[j].
    """
    if f_true is None:
        f_true = get_default_f_true

    cache_key = (j, L)

    if cache_key not in _F_HAT_CACHE:
        if abs(L - np.pi) < 1e-10:  # Equal to pi

            def integrand(x: float) -> float:
                return float(f_true(x) * np.sin(j * x))
        else:

            def integrand(x: float) -> float:
                return float(f_true(x) * np.sin(j * np.pi * x / L))

        result, _ = quad(integrand, 0, L)
        _F_HAT_CACHE[cache_key] = result * 2 / L

    return float(_F_HAT_CACHE[cache_key])


def compute_t_vec(n: int, T: float) -> np.ndarray:
    """
    Compute the time vectors for modified factorial sequence t_j.

    Args:
        n: Number of time points.
        T: Time scaling parameter.

    Returns:
        Array of n time values.
    """

    def _compute_t_vec_optimized(n: int, T: float) -> np.ndarray:
        """Optimized computation of time vector."""
        t_vec = np.zeros(n)
        fact_cache = np.ones(2 * n + 1, dtype=np.float64)  # Store factorials

        for i in range(1, 2 * n + 1):
            fact_cache[i] = fact_cache[i - 1] * i  # 2n! part of factorial time sequence

        for k in range(1, n + 1):
            numerator = fact_cache[2 * k - 1]
            denominator = (8.0 ** (k - 1)) * fact_cache[k] * fact_cache[k - 1]
            t_vec[k - 1] = (numerator / denominator) * T

        return t_vec

    # Compute time vectors with or without numba
    if NUMBA_AVAILABLE:
        _compute_t_vec_optimized = jit(nopython=True, cache=True)(
            _compute_t_vec_optimized
        )
        t_vec = _compute_t_vec_optimized(n, T)
    else:
        t_vec = _compute_t_vec_optimized(n, T)

    return t_vec
