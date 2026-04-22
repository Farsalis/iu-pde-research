"""Forcing term computations for non-homogeneous PDE.

Computes the Fourier coefficient of the forcing term F(x, t) for a given mode,
Holthe contribution of the forcing term to the solution at a given time, and the
truncation index n_k for the forcing series (non-homogeneous heat equation with
Dirichlet boundary conditions).
"""

import math
from collections.abc import Callable

import numpy as np
from numpy.polynomial import legendre
from scipy.integrate import quad

# Fixed order for inner x-integral (Fourier mode integrand); avoids adaptive cost.
_F_HAT_J_QUAD_ORDER = 32
# Number of time samples for precomputed F_hat_j in forcing contribution (no nested quad).
_FORCING_S_GRID_SIZE = 24
# Tolerance for exponential decay cutoff: e^{-j^2 t_k} < _EXP_DECAY_TOL triggers the
# cheap F_hat_j(t_k)/j^2 approximation instead of full numerical integration.
_EXP_DECAY_TOL = 1e-12


def compute_exp_decay_cutoff(
    tk: float,
    L: float = np.pi,
    tol: float = _EXP_DECAY_TOL,
) -> int:
    """
    Compute the frequency cutoff N where e^{-lambda_j * t_k} drops below *tol*.

    For j > N the exponential kernel in the forcing integral is negligible, so
    the integral can be replaced by the cheap approximation F_hat_j(t_k) / lambda_j.
    The eigenvalue is lambda_j = j^2 for L = pi, or (j*pi/L)^2 in general.

    Args:
        tk: Current time value (must be > 0 for a meaningful cutoff).
        L: Domain length. Defaults to pi.
        tol: Decay threshold. Defaults to 1e-12 (12 significant figures).

    Returns:
        Integer cutoff N (>= 0). Modes j = 1..N need the full integral;
        modes j = N+1..n_k can use the asymptotic approximation.
    """
    if tk <= 0 or tol <= 0 or tol >= 1:
        return 0
    log_inv_tol = -math.log(tol)
    eigenvalue_scale = (np.pi / L) ** 2
    return math.ceil(math.sqrt(log_inv_tol / (eigenvalue_scale * tk)))


def compute_forcing_truncation_constant_C(
    F_x: Callable[[float, float], float],
    T: float,
    L: float = np.pi,
    *,
    n_s_samples: int = 50,
) -> float:
    """
    Compute the constant C used in the definition of the truncation index n_k.

    C = (2/L) * max_{0 <= s <= T} ||F_x(·, s)||_{L^1(0,L)}, where F_x denotes
    the spatial partial derivative of the forcing term F.

    Args:
        F_x: Spatial derivative of forcing, F_x(x, s) = ∂F/∂x(x, s).
        T: Time horizon; maximum over s is taken on [0, T].
        L: Domain length. Defaults to pi.
        n_s_samples: Number of time samples for the max over s. Defaults to 50.

    Returns:
        The constant C (non-negative float).
    """
    if T <= 0 or n_s_samples < 1:
        return 0.0

    s_grid = np.linspace(0, T, n_s_samples)
    max_L1 = 0.0

    for s in s_grid:
        s_val = float(s)

        def integrand(x: float, s_opt: float = s_val) -> float:
            return float(abs(F_x(x, s_opt)))

        integral_val, _ = quad(integrand, 0, L)
        norm_L1 = integral_val
        if norm_L1 > max_L1:
            max_L1 = norm_L1

    return float((2.0 / L) * max_L1)


def compute_n_k(k: int, t_k: float, C: float) -> int:
    """
    Compute the truncation index n_k for the forcing series at step k.

    n_k = ceil(C * exp((k+1)^2 * t_k / 2)), so that the tail of the source
    term is negligible relative to the reconstruction error.

    Args:
        k: Step index (1-based).
        t_k: Time value at step k.
        C: Constant from compute_forcing_truncation_constant_C.

    Returns:
        Integer truncation index for use in loops (sum j=1..n_k).
    """
    if C <= 0 or t_k < 0:
        return 0
    exponent = (k + 1) ** 2 * t_k / 2.0
    raw = C * math.exp(exponent)
    return max(0, math.ceil(raw))


def _fixed_quad_x(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Integrate f over [a, b] using n-point Gauss-Legendre (no adaptive calls)."""
    xi, w = legendre.leggauss(n)
    # Map [-1, 1] -> [a, b]: x = (b - a) / 2 * xi + (a + b) / 2, dx = (b - a) / 2 * dxi
    x_nodes = (b - a) / 2.0 * xi + (a + b) / 2.0
    half_len = (b - a) / 2.0
    return float(half_len * np.sum(w * np.array([f(x) for x in x_nodes])))


# Compute synthetic measurement data for the forcing term.
def compute_F_hat_j(
    F: Callable[[float, float], float],
    j: int,
    t: float,
    L: float = np.pi,
) -> float:
    """
    Compute Fourier coefficient of forcing term F(x, t) for mode j.

    Uses fixed-order Gauss-Legendre quadrature for predictable, fast integration.

    Args:
        F: Forcing function F(x, t).
        j: Mode number (positive integer).
        t: Time value.
        L: Domain length. Defaults to pi.

    Returns:
        Fourier coefficient F_hat[j] at time t.
    """
    if abs(L - np.pi) < 1e-10:

        def integrand(x: float) -> float:
            return float(F(x, t) * np.sin(j * x))
    else:

        def integrand(x: float) -> float:
            return float(F(x, t) * np.sin(j * np.pi * x / L))

    result = _fixed_quad_x(integrand, 0.0, float(L), _F_HAT_J_QUAD_ORDER)
    return float(float(result * 2.0 / L))


def compute_forcing_contribution_at_time(
    F: Callable[[float, float], float],
    tk: float,
    x0: float,
    n_terms: int = 50,
    L: float = np.pi,
) -> float:
    """
    Compute forcing term contribution to solution at time tk.

    Uses a two-strategy summation split at an adaptive cutoff N determined by
    exponential decay of the heat kernel:

    * j = 1..N  (full integral): precompute F_hat_j on a time grid and
      integrate via the trapezoidal rule.
    * j = N+1..n_terms  (asymptotic): approximate the integral by
      F_hat_j(t_k) / lambda_j, valid once e^{-lambda_j * t_k} is negligible.

    Args:
        F: Forcing function F(x, t).
        tk: Time value.
        x0: Spatial measurement point.
        n_terms: Number of Fourier modes (n_k, possibly capped). Defaults to 50.
        L: Domain length. Defaults to pi.

    Returns:
        Contribution of forcing term to u(x0, tk).
    """
    if tk <= 0 or n_terms < 1:
        return 0.0

    use_pi_formulation = abs(L - np.pi) < 1e-10

    # --- Adaptive cutoff ------------------------------------------------
    n_cutoff = compute_exp_decay_cutoff(tk, L)
    n_full = min(n_cutoff, n_terms)

    # --- Strategy 1: full integral for low-frequency modes (j=1..n_full) -
    sum_val = 0.0
    if n_full > 0:
        n_s = _FORCING_S_GRID_SIZE
        s_grid = np.linspace(0.0, float(tk), n_s)

        F_hat_grid = np.zeros((n_full, n_s))
        for j in range(1, n_full + 1):
            for i_s, s in enumerate(s_grid):
                F_hat_grid[j - 1, i_s] = compute_F_hat_j(F, j, float(s), L)

        for j in range(1, n_full + 1):
            j_sq: float = float(j**2) if use_pi_formulation else (j * np.pi / L) ** 2
            exponent = -j_sq * (tk - s_grid)
            exponent = np.clip(exponent, None, 700.0)
            vals = F_hat_grid[j - 1, :] * np.exp(exponent)
            integral_j = float(np.trapezoid(vals, s_grid))
            if use_pi_formulation:
                sum_val += integral_j * np.sin(j * x0)
            else:
                sum_val += integral_j * np.sin(j * np.pi * x0 / L)

    # --- Strategy 2: asymptotic approx for high-frequency modes ----------
    for j in range(n_full + 1, n_terms + 1):
        f_hat_j_tk = compute_F_hat_j(F, j, float(tk), L)
        if use_pi_formulation:
            lambda_j = float(j**2)
            sin_j_x0 = float(np.sin(j * x0))
        else:
            lambda_j = float((j * np.pi / L) ** 2)
            sin_j_x0 = float(np.sin(j * np.pi * x0 / L))
        sum_val += (f_hat_j_tk / lambda_j) * sin_j_x0

    return float(sum_val)
