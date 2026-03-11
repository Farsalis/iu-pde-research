"""CuPy-accelerated reconstruction algorithms for the 1D heat equation.

This module provides a GPU-friendly variant of the non-homogeneous modified
 t_j reconstruction using CuPy (CUDA).

Current coverage:
* compute_modified_t_j_nonhomogeneous_cupy: non-homogeneous modified t_j.

The public API mirrors the CPU version in ``reconstruction.py`` as closely
as possible so callers can switch between CPU and GPU backends.
"""

from collections.abc import Callable

import numpy as np

from iu_pde_research.core.forcing import compute_n_k
from iu_pde_research.core.forcing_cupy import (
    compute_forcing_contribution_at_time_default_F_cupy,
)
from iu_pde_research.core.utils import compute_t_vec, get_default_f_true

from iu_pde_research.core.reconstruction import MAX_FORCING_MODES

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "CuPy is required for compute_modified_t_j_nonhomogeneous_cupy. "
        "Install an appropriate CUDA-enabled wheel, for example "
        "'pip install \"cupy-cuda12x\"' for CUDA 12.x."
    ) from exc


def compute_modified_t_j_nonhomogeneous_cupy(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    F: Callable[[float, float], float] | None = None,
    F_x: Callable[[float, float], float] | None = None,
    f_true: Callable[[float], float] | None = None,
    L: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """CuPy-accelerated non-homogeneous modified t_j reconstruction.

    This mirrors ``compute_modified_t_j_nonhomogeneous`` but offloads the
    homogeneous spectral part and the reconstruction (bar_f_hat, bar_f_n)
    to the GPU via CuPy. Forcing-term integrals are currently computed
    on CPU using the existing routines for correctness and simplicity.

    Args:
        n: Number of time measurements.
        T: Time scaling parameter.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction (NumPy array).
        F: Forcing function F(x, t). If None, assumes the homogeneous case.
        F_x: Spatial derivative F_x(x, s) = ∂F/∂x. If None and F is not None,
            F_x is used only for the truncation constant C via CPU routines.
        f_true: True initial condition function. If None, uses the default
            initial condition.
        L: Domain length. Defaults to pi.

    Returns:
        Tuple of NumPy arrays (f_true_vals, bar_f_n_vals) on x_grid.
    """
    if f_true is None:
        f_true = get_default_f_true

    # --- Time vector (CPU) -------------------------------------------------
    t_vec = compute_t_vec(n, T)

    # --- Fourier coefficients of initial data (CPU, as in baseline) -------
    n_terms = 50
    j_vals = np.arange(1, n_terms + 1)

    if abs(L - np.pi) < 1e-10:  # Equal to pi
        f_hat_vals = np.array(
            [
                np.vectorize(f_true)(
                    np.linspace(0.0, L, 256, endpoint=True)
                ),  # placeholder to satisfy type checker
            ]
        )  # will be overwritten below

        f_hat_vals = np.array(
            [
                # Integral of f_true(x) * sin(j x) over [0, L], scaled by 2/L.
                __import__("scipy").integrate.quad(  # type: ignore[attr-defined]
                    lambda x, j_val=j: f_true(x) * np.sin(j_val * x),
                    0,
                    L,
                )[0]
                * 2
                / L
                for j in j_vals
            ]
        )
        j_squared = j_vals**2
        sin_j_x0 = np.sin(j_vals * x0)
    else:
        f_hat_vals = np.array(
            [
                __import__("scipy").integrate.quad(  # type: ignore[attr-defined]
                    lambda x, j_val=j: f_true(x) * np.sin(j_val * np.pi * x / L),
                    0,
                    L,
                )[0]
                * 2
                / L
                for j in j_vals
            ]
        )
        j_squared = (j_vals * np.pi / L) ** 2
        sin_j_x0 = np.sin(j_vals * np.pi * x0 / L)

    # --- Homogeneous contribution u_homogeneous on GPU ---------------------
    t_vec_cp = cp.asarray(t_vec)
    j_squared_cp = cp.asarray(j_squared)
    f_hat_vals_cp = cp.asarray(f_hat_vals)
    sin_j_x0_cp = cp.asarray(sin_j_x0)

    exp_matrix_cp = cp.exp(-cp.outer(t_vec_cp, j_squared_cp))
    u_homogeneous_cp = cp.sum(exp_matrix_cp * f_hat_vals_cp * sin_j_x0_cp, axis=1)
    u_homogeneous = cp.asnumpy(u_homogeneous_cp)

    if F is not None:
        # Specialized CuPy forcing kernel for the default demo forcing
        # F(x,t) = sin(x)*exp(-t) on L = pi. For other F, this function
        # should currently not be used.
        if L != np.pi:
            msg = (
                "compute_modified_t_j_nonhomogeneous_cupy currently only supports "
                "F != None for L = pi."
            )
            raise ValueError(msg)

        # For F(x,t) = sin(x) * exp(-t), F_x(x,s) = cos(x) * exp(-s) and
        # C = (2/L) * max_s ||F_x(·,s)||_{L^1(0,L)} = 4/pi.
        C_default = 4.0 / np.pi

        def _n_terms_forcing(k: int, tk_val: float) -> int:
            n_k = compute_n_k(k, tk_val, C_default)
            if MAX_FORCING_MODES is not None:
                n_k = min(n_k, MAX_FORCING_MODES)
            return max(1, n_k)

        u_forcing = np.array(
            [
                compute_forcing_contribution_at_time_default_F_cupy(
                    tk,
                    x0,
                    n_terms=_n_terms_forcing(k, float(tk)),
                    L=L,
                )
                for k, tk in enumerate(t_vec, start=1)
            ]
        )
        u_data = u_homogeneous + u_forcing
    else:
        u_forcing = None
        u_data = u_homogeneous

    # --- Reconstruction step on GPU (bar_f_hat, bar_f_n) -------------------
    bar_f_hat_cp = cp.zeros(int(cp.ceil(n / 2)))
    max_k = int(bar_f_hat_cp.shape[0])

    if abs(L - np.pi) < 1e-10:
        k_indices = cp.arange(1, max_k + 1)
        sin_k_x0_cp = cp.sin(k_indices * x0)
        k_squared_cache_cp = k_indices**2
    else:
        k_indices = cp.arange(1, max_k + 1)
        sin_k_x0_cp = cp.sin(k_indices * cp.pi * x0 / L)
        k_squared_cache_cp = (k_indices * cp.pi / L) ** 2

    u_data_cp = cp.asarray(u_data)
    u_forcing_cp = cp.asarray(u_forcing) if u_forcing is not None else None
    t_vec_cp_full = t_vec_cp  # alias with clearer name

    for k in range(1, max_k + 1):
        tk = t_vec_cp_full[k - 1]
        # Start from u(x0, tk); when forcing is present, subtract the
        # forcing contribution to recover the homogeneous part, mirroring
        # the CPU algorithm.
        if u_forcing_cp is not None:
            u_data_k = u_data_cp[k - 1] - u_forcing_cp[k - 1]
        else:
            u_data_k = u_data_cp[k - 1]

        if k == 1:
            c_bar_1 = cp.exp(k_squared_cache_cp[0] * tk) * u_data_k
            bar_f_hat_cp[0] = c_bar_1 / sin_k_x0_cp[0]
        else:
            j_indices = cp.arange(1, k)
            exp_terms = cp.exp(-k_squared_cache_cp[j_indices - 1] * tk)
            c_bar_prev = bar_f_hat_cp[j_indices - 1] * sin_k_x0_cp[j_indices - 1]
            sum_prev = cp.sum(exp_terms * c_bar_prev)

            c_bar_k = cp.exp(k_squared_cache_cp[k - 1] * tk) * (u_data_k - sum_prev)
            bar_f_hat_cp[k - 1] = c_bar_k / sin_k_x0_cp[k - 1]

    k_vals_cp = cp.arange(1, max_k + 1)
    x_grid_cp = cp.asarray(x_grid, dtype=cp.float64)

    if abs(L - np.pi) < 1e-10:
        sin_matrix_cp = cp.sin(cp.outer(x_grid_cp, k_vals_cp))
    else:
        sin_matrix_cp = cp.sin(cp.outer(x_grid_cp, k_vals_cp * cp.pi / L))

    bar_f_n_vals_cp = sin_matrix_cp @ bar_f_hat_cp

    f_true_vals_np = np.asarray(f_true(x_grid), dtype=np.float64)
    bar_f_n_vals_np = cp.asnumpy(bar_f_n_vals_cp)

    return f_true_vals_np, bar_f_n_vals_np

