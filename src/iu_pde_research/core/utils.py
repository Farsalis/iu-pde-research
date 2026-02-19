"""Utility functions and constants for PDE reconstruction."""

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

# Allows to compile factorials just in time.
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Dummy decorator when numba is not available."""

        def decorator(func):
            return func

        return decorator

    prange = range

# Cache for Fourier coefficients
_F_HAT_CACHE = {}


def get_default_f_true(x: float) -> float:
    """
    Default initial condition function for the homogeneous heat equation.

    :args:
        x: Spatial coordinate.

    :returns:
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

    :args:
        j: Mode number (positive integer).
        f_true: Initial condition function. If None, uses default.
        L: Domain length. Defaults to pi.

    :returns:
        Fourier coefficient f_hat[j].
    """
    if f_true is None:
        f_true = get_default_f_true

    cache_key = (j, L)

    if cache_key not in _F_HAT_CACHE:
        if abs(L - np.pi) < 1e-10:  # Equal to pi
            integrand = lambda x: f_true(x) * np.sin(j * x)
        else:
            integrand = lambda x: f_true(x) * np.sin(j * np.pi * x / L)

        result, _ = quad(integrand, 0, L)
        _F_HAT_CACHE[cache_key] = result * 2 / L

    return _F_HAT_CACHE[cache_key]


def compute_t_vec(n: int, T: float) -> np.ndarray:
    """
    Compute the time vectors for modified factorial sequence t_j.

    :args:
        n: Number of time points.
        T: Time scaling parameter.

    :returns:
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
