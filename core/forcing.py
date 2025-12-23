"""Forcing term computations for non-homogeneous PDE."""

import numpy as np
import jax.numpy as jnp
from scipy.integrate import quad
from scipy.optimize import minimize
from typing import Callable, Optional
from jax import grad

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

# TODO: Properly implement n_k to avoid overflow.
"""
def compute_c_k(F:Callable[[float, float], float], tk:float, x0:float, k: int):

    max_F_s = []

    def integrand(s):
        return lambda j: compute_F_hat_j(F, j, s)
    
    for m in range(1, k+1):


    def compute_F_xs():
        if s < 0 or s > tk:
            return 0.0
        else: 
            # F_x = grad(F, argnums=0)  # Partial derivative with respect to x of F(x, t // s).
            # return F_x(x0, s)
            pass

        integral_val = d

        

            
    max_F_s =  minimize(compute_F_x, bounds=(0, tk), method="bounded")

    print(max_F_s.fun, max_F_s.x)



    return (1/np.pi) * max_F_s.fun
"""
    

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

#  Prevent overflow due to large exponents.
    for j in j_vals:

        def integrand(s):
            if s < 0 or s > tk:
                return 0.0
            F_hat_j = compute_F_hat_j(F, j, s, L)

            if use_pi_formulation:  # 
                j_sq = j**2  # Simplified form for pi case
                exponent = -j_sq * (tk - s)  # Exponents in the j-summations from Green's function
                if exponent > 700:  # Found through trial and error
                    return 0.0
                return F_hat_j * np.exp(exponent)
            else:
                j_sq = (j * np.pi / L)**2  # From gamma during derivation
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
            sum_val += integral_val * np.sin(j * x0)  # Outside sine term in summation
        else:
            sum_val += integral_val * np.sin(j * np.pi * x0 / L)

    return sum_val

