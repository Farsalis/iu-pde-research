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

# Allows to compile factorials just in time.
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
    return np.sin(2 * x) + 0.5 * np.sin(5 * x)


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
            denominator = (8 ** (k - 1)) * fact_cache[k] * fact_cache[k - 1]
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
