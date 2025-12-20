"""Forcing term computations for non-homogeneous PDE."""

import numpy as np
from scipy.integrate import quad
from typing import Callable, Optional

#  Compute synthetic measurement data for the forcing term.
def compute_F_hat_j(
    F: Callable[[float, float], float],
    j: int,
    t: float,
    L: float = np.pi,
) -> float:
    """
    Compute Fourier coefficient of forcing term F(x,t) for mode j.

    Args:
        F: Forcing function F(x, t).
        j: Mode number (positive integer).
        t: Time value.
        L: Domain length. Defaults to pi.

    Returns:
        Fourier coefficient F_hat[j] at time t.
    """
    if abs(L - np.pi) < 1e-10:
        integrand = lambda x: F(x, t) * np.sin(j * x)
    else:
        integrand = lambda x: F(x, t) * np.sin(j * np.pi * x / L)

    result, _ = quad(integrand, 0, L)

    return result * 2 / L


def compute_forcing_contribution_at_time(
    F: Callable[[float, float], float],
    tk: float,
    x0: float,
    n_terms: int = 50,
    L: float = np.pi,
) -> float:
    """
    Compute forcing term contribution to solution at time tk.

    Args:
        F: Forcing function F(x, t).
        tk: Time value.
        x0: Spatial measurement point.
        n_terms: Number of Fourier modes to use. Defaults to 50.
        L: Domain length. Defaults to pi.

    Returns:
        Contribution of forcing term to u(x0, tk).
    """
    sum_val = 0.0
    j_vals = np.arange(1, n_terms + 1)
    use_pi_formulation = abs(L - np.pi) < 1e-10

#  TODO: Implement recursive relation n_k to avoid overflow.

#  Prevent overflow due to large exponents.
    for j in j_vals:

        def integrand(s):
            if s < 0 or s > tk:
                return 0.0
            F_hat_j = compute_F_hat_j(F, j, s, L)

            if use_pi_formulation:
                j_sq = j**2
                exponent = -j_sq * (tk - s)
                if exponent > 700:
                    return 0.0
                return F_hat_j * np.exp(exponent)
            else:
                j_sq = (j * np.pi / L)**2
                exponent = -j_sq * (tk - s)
                if exponent > 700:
                    return 0.0
                return F_hat_j * np.exp(exponent)

        try:
            integral_val, _ = quad(
                integrand, 0, tk, limit=200, epsabs=1e-10, epsrel=1e-10
            )
        except (OverflowError, ValueError):
            integral_val = 0.0

        if use_pi_formulation:
            sum_val += integral_val * np.sin(j * x0)
        else:
            sum_val += integral_val * np.sin(j * np.pi * x0 / L)

    return sum_val

