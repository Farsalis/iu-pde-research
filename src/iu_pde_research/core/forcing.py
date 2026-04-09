"""Forcing term computations for non-homogeneous PDE.

Computes the Fourier coefficient of the forcing term F(x, t) for a given mode,
and the contribution of the forcing term to the solution at a given time
(non-homogeneous heat equation with Dirichlet boundary conditions).
"""

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad


# Compute synthetic measurement data for the forcing term.
def compute_F_hat_j(
    F: Callable[[float, float], float],
    j: int,
    t: float,
    L: float = np.pi,
) -> float:
    """
    Compute Fourier coefficient of forcing term F(x,t) for mode j.

    :args:
        F: Forcing function F(x, t).
        j: Mode number (positive integer).
        t: Time value.
        L: Domain length. Defaults to pi.

    :returns:
        Fourier coefficient F_hat[j] at time t.
    """
    if abs(L - np.pi) < 1e-10:
        integrand = lambda x: F(x, t) * np.sin(j * x)
    else:
        integrand = lambda x: F(x, t) * np.sin(j * np.pi * x / L)

    result, _ = quad(integrand, 0, L)

    return float(result * 2 / L)


def compute_forcing_contribution_at_time(
    F: Callable[[float, float], float],
    tk: float,
    x0: float,
    n_terms: int = 50,
    L: float = np.pi,
) -> float:
    """
    Compute forcing term contribution to solution at time tk.

    :args:
        F: Forcing function F(x, t).
        tk: Time value.
        x0: Spatial measurement point.
        n_terms: Number of Fourier modes to use. Defaults to 50.
        L: Domain length. Defaults to pi.

    :returns:
        Contribution of forcing term to u(x0, tk).
    """
    sum_val = 0.0
    j_vals = np.arange(1, n_terms + 1)
    use_pi_formulation = abs(L - np.pi) < 1e-10

    # Prevent overflow due to large exponents.
    for j in j_vals:
        j_mode = int(j)

        def integrand(s: float) -> float:
            if s < 0 or s > tk:
                return 0.0
            F_hat_j = compute_F_hat_j(F, j_mode, s, L)

            if use_pi_formulation:
                j_sq = float(j_mode**2)  # Simplified form for pi case
                exponent = -j_sq * (
                    tk - s
                )  # Exponents in the j-summations from Green's function
                if exponent > 700:  # Found through trial and error
                    return 0.0
                return float(F_hat_j * np.exp(exponent))
            else:
                j_sq = (j_mode * np.pi / L) ** 2  # From gamma during derivation
                exponent = -j_sq * (tk - s)
                if exponent > 700:
                    return 0.0
                return float(F_hat_j * np.exp(exponent))

        try:
            integral_val, _ = quad(
                integrand, 0, tk, limit=200, epsabs=1e-10, epsrel=1e-10
            )
        except (OverflowError, ValueError):
            integral_val = 0.0

        if use_pi_formulation:
            sum_val += integral_val * np.sin(
                j_mode * x0
            )  # Outside sine term in summation
        else:
            sum_val += integral_val * np.sin(j_mode * np.pi * x0 / L)

    return float(sum_val)
