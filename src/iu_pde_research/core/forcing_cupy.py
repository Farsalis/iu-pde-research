"""CuPy-accelerated forcing computations for the non-homogeneous heat equation.

This module provides GPU kernels for:

* A specialized default forcing term used in the CLI demo:
  F(x, t) = sin(x) * exp(-t), L = pi.
* General CuPy-based forcing and derivative callables, suitable for use in
  the main non-homogeneous reconstruction with backend=\"gpu\".
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "CuPy is required for forcing_cupy. "
        "Install an appropriate CUDA-enabled wheel, for example "
        "'pip install \"cupy-cuda12x[ctk]\"' for CUDA 12.x."
    ) from exc


def default_F_cupy(x: cp.ndarray, t: cp.ndarray) -> cp.ndarray:
    """Default forcing term F(x,t)=sin(x)*exp(-t) implemented with CuPy ops."""
    return cp.sin(x) * cp.exp(-t)


def default_F_x_cupy(x: cp.ndarray, t: cp.ndarray) -> cp.ndarray:
    """Spatial derivative F_x(x,t)=cos(x)*exp(-t) implemented with CuPy ops."""
    return cp.cos(x) * cp.exp(-t)


def compute_forcing_truncation_constant_C_gpu(
    F_x_gpu: Callable[[cp.ndarray, cp.ndarray], cp.ndarray],
    T: float,
    L: float = np.pi,
    *,
    n_x: int = 512,
    n_s: int = 64,
) -> float:
    """GPU analogue of compute_forcing_truncation_constant_C.

    Computes
        C = (2/L) * max_{0 <= s <= T} ||F_x(·, s)||_{L^1(0,L)}

    using fixed grids in x and s and CuPy-based trapezoidal quadrature.
    """
    if T <= 0 or n_s < 1 or n_x < 2:
        return 0.0

    x_grid = cp.linspace(0.0, float(L), int(n_x))
    s_grid = cp.linspace(0.0, float(T), int(n_s))

    x2d = x_grid[None, :]  # (1, Nx)
    s2d = s_grid[:, None]  # (Ns, 1)

    vals = cp.abs(F_x_gpu(x2d, s2d))  # (Ns, Nx)
    L1_vals = cp.trapezoid(vals, x_grid, axis=1)  # (Ns,)
    max_L1 = cp.max(L1_vals)

    C_gpu = (2.0 / float(L)) * max_L1
    return float(C_gpu.get())


def compute_forcing_contribution_at_time_default_F_cupy(
    tk: float,
    x0: float,
    n_terms: int = 50,
    *,
    L: float = np.pi,
    n_x: int = 512,
    n_s: int = 64,
) -> float:
    """GPU forcing contribution at x0, tk for F(x,t)=sin(x)*exp(-t), L=pi.

    Approximates

        u_F(x0, tk) = sum_{j=1}^{n_terms} ∫_0^{tk} F_hat_j(s) e^{-lambda_j (tk-s)} ds * sin(j x0),

    where

        F_hat_j(s) = (2/L) ∫_0^L F(x,s) sin(j x) dx,
        lambda_j = j^2  (for L=pi).

    The integrals over x and s are computed via fixed-grid trapezoidal rules
    on the GPU using CuPy, batching all modes j and time samples s.
    """
    if tk <= 0.0 or n_terms < 1:
        return 0.0

    if abs(L - math.pi) > 1e-10:
        msg = "compute_forcing_contribution_at_time_default_F_cupy assumes L = pi."
        raise ValueError(msg)

    # === Grids on GPU ===
    x_grid = cp.linspace(0.0, float(L), int(n_x))
    s_grid = cp.linspace(0.0, float(tk), int(n_s))  # integration grid in time

    # === F(x,s) on (s,x) grid ===
    x2d = x_grid[None, :]  # (1, Nx)
    s2d = s_grid[:, None]  # (Ns, 1)
    F_vals = default_F_cupy(x2d, s2d)  # (Ns, Nx)

    # === Precompute sin(j x) over x for all j ===
    j_vals = cp.arange(1, int(n_terms) + 1, dtype=cp.float64)  # (J,)
    sin_jx = cp.sin(j_vals[:, None] * x_grid[None, :])  # (J, Nx)

    # === Compute F_hat_j(s) for all j and s ===
    # integrand_x: (Ns, J, Nx)
    integrand_x = F_vals[:, None, :] * sin_jx[None, :, :]
    # Trapezoid over x -> (Ns, J)
    F_hat = cp.trapezoid(integrand_x, x_grid, axis=2) * (2.0 / float(L))

    # === Integrate over s with heat-kernel factor for each j ===
    lam_j = j_vals**2  # (J,)

    # exponent: (Ns, J)
    exponent = -lam_j[None, :] * (float(tk) - s_grid[:, None])
    kernel = cp.exp(exponent)

    integrand_s = F_hat * kernel  # (Ns, J)
    # Trapezoid over s -> (J,)
    integral_j = cp.trapezoid(integrand_s, s_grid, axis=0)

    # === Sum over j with sin(j x0) ===
    sin_j_x0 = cp.sin(j_vals * float(x0))  # (J,)
    u_F_tk = cp.sum(integral_j * sin_j_x0)

    return float(u_F_tk.get())


def compute_forcing_contribution_at_time_gpu(
    F_gpu: Callable[[cp.ndarray, cp.ndarray], cp.ndarray],
    tk: float,
    x0: float,
    n_terms: int,
    *,
    L: float = np.pi,
    n_x: int = 512,
    n_s: int = 64,
) -> float:
    """General GPU forcing contribution at x0, tk for a CuPy-based F_gpu.

    This is the generalization of
    compute_forcing_contribution_at_time_default_F_cupy to arbitrary
    CuPy-compatible F_gpu(x, t).
    """
    if tk <= 0.0 or n_terms < 1:
        return 0.0

    x_grid = cp.linspace(0.0, float(L), int(n_x))
    s_grid = cp.linspace(0.0, float(tk), int(n_s))

    x2d = x_grid[None, :]  # (1, Nx)
    s2d = s_grid[:, None]  # (Ns, 1)
    F_vals = F_gpu(x2d, s2d)  # (Ns, Nx)

    j_vals = cp.arange(1, int(n_terms) + 1, dtype=cp.float64)  # (J,)
    if abs(L - math.pi) < 1e-10:
        sin_jx = cp.sin(j_vals[:, None] * x_grid[None, :])  # (J, Nx)
        lam_j = j_vals**2
        sin_j_x0 = cp.sin(j_vals * float(x0))
    else:
        sin_jx = cp.sin(j_vals[:, None] * cp.pi * x_grid[None, :] / float(L))
        lam_j = (j_vals * cp.pi / float(L)) ** 2
        sin_j_x0 = cp.sin(j_vals * cp.pi * float(x0) / float(L))

    integrand_x = F_vals[:, None, :] * sin_jx[None, :, :]  # (Ns, J, Nx)
    F_hat = cp.trapezoid(integrand_x, x_grid, axis=2) * (2.0 / float(L))  # (Ns, J)

    exponent = -lam_j[None, :] * (float(tk) - s_grid[:, None])  # (Ns, J)
    kernel = cp.exp(exponent)

    integrand_s = F_hat * kernel  # (Ns, J)
    integral_j = cp.trapezoid(integrand_s, s_grid, axis=0)  # (J,)

    u_F_tk = cp.sum(integral_j * sin_j_x0)
    return float(u_F_tk.get())

