"""CuPy-accelerated reconstruction algorithms for the 1D heat equation.

This module provides a GPU-friendly variant of the non-homogeneous modified
 t_j reconstruction using CuPy (CUDA).

Current coverage:
* compute_modified_t_j_nonhomogeneous_cupy: non-homogeneous modified t_j.

The public API mirrors the CPU version in ``reconstruction.py`` as closely
as possible so callers can switch between CPU and GPU backends.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from iu_pde_research.core.forcing import compute_n_k
from iu_pde_research.core.forcing_cupy import (
    compute_forcing_contribution_at_time_gpu,
    compute_forcing_truncation_constant_C_gpu,
    default_F_cupy,
    default_F_x_cupy,
)
from iu_pde_research.core.reconstruction import MAX_FORCING_MODES
from iu_pde_research.core.utils import (
    InitialCondition,
    compute_t_vec,
    get_default_f_true,
)

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "CuPy is required for GPU-accelerated non-homogeneous reconstruction. "
        "Install an appropriate CUDA-enabled wheel, for example "
        "'pip install \"cupy-cuda12x\"' for CUDA 12.x."
    ) from exc


def _compute_homogeneous_gpu(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    f_true: InitialCondition | None = None,
    L: float = np.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared homogeneous part for GPU path: returns (t_vec, u_homogeneous_cp, j_vals, k_vals)."""
    if f_true is None:
        f_true = get_default_f_true

    t_vec = compute_t_vec(n, T)

    n_terms = 50
    j_vals = np.arange(1, n_terms + 1)

    j_squared: np.ndarray[tuple[Any, ...], np.dtype[np.float64]]

    if abs(L - np.pi) < 1e-10:
        f_hat_vals = np.array(
            [
                __import__("scipy").integrate.quad(
                    lambda x, j_val=j: f_true(x) * np.sin(j_val * x),
                    0,
                    L,
                )[0]
                * 2
                / L
                for j in j_vals
            ]
        )
        j_squared = (j_vals**2).astype(np.float64)
        sin_j_x0 = np.sin(j_vals * x0)
    else:
        f_hat_vals = np.array(
            [
                __import__("scipy").integrate.quad(
                    lambda x, j_val=j: f_true(x) * np.sin(j_val * np.pi * x / L),
                    0,
                    L,
                )[0]
                * 2
                / L
                for j in j_vals
            ]
        )
        j_squared = ((j_vals * np.pi / L) ** 2).astype(np.float64)
        sin_j_x0 = np.sin(j_vals * np.pi * x0 / L)

    t_vec_cp = cp.asarray(t_vec)
    j_squared_cp = cp.asarray(j_squared)
    f_hat_vals_cp = cp.asarray(f_hat_vals)
    sin_j_x0_cp = cp.asarray(sin_j_x0)

    exp_matrix_cp = cp.exp(-cp.outer(t_vec_cp, j_squared_cp))
    u_homogeneous_cp = cp.sum(exp_matrix_cp * f_hat_vals_cp * sin_j_x0_cp, axis=1)

    return t_vec, u_homogeneous_cp, j_vals, np.arange(1, int(cp.ceil(n / 2)) + 1)


def compute_modified_t_j_nonhomogeneous_gpu(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    F_gpu: Callable[[cp.ndarray, cp.ndarray], cp.ndarray],
    F_x_gpu: Callable[[cp.ndarray, cp.ndarray], cp.ndarray],
    f_true: InitialCondition | None = None,
    L: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """General CuPy-accelerated non-homogeneous modified t_j reconstruction."""
    if f_true is None:
        f_true = get_default_f_true

    t_vec, u_homogeneous_cp, _, _ = _compute_homogeneous_gpu(
        n=n, T=T, x0=x0, x_grid=x_grid, f_true=f_true, L=L
    )

    # Compute truncation constant C on GPU from F_x_gpu.
    C = compute_forcing_truncation_constant_C_gpu(F_x_gpu, T=T, L=L)

    # Per-t_k forcing contribution using general GPU kernel.
    def _n_terms_forcing(k: int, tk_val: float) -> int:
        n_k = compute_n_k(k, tk_val, C)
        if MAX_FORCING_MODES is not None:
            n_k = min(n_k, MAX_FORCING_MODES)
        return max(1, n_k)

    u_forcing = np.array(
        [
            compute_forcing_contribution_at_time_gpu(
                F_gpu,
                tk,
                x0,
                n_terms=_n_terms_forcing(k, float(tk)),
                L=L,
            )
            for k, tk in enumerate(t_vec, start=1)
        ]
    )

    u_data_cp = u_homogeneous_cp + cp.asarray(u_forcing)

    # Reconstruction loop is identical in structure to the CPU path,
    # but executed on GPU.
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

    t_vec_cp = cp.asarray(t_vec)

    for k in range(1, max_k + 1):
        tk = t_vec_cp[k - 1]
        u_data_k = u_data_cp[k - 1]  # already includes forcing

        if k == 1:
            # Subtract forcing to recover homogeneous part, as in CPU algorithm.
            n_k = _n_terms_forcing(k, float(tk))
            forcing_contrib = compute_forcing_contribution_at_time_gpu(
                F_gpu, float(tk), x0, n_terms=n_k, L=L
            )
            u_data_k = u_data_k - forcing_contrib

            c_bar_1 = cp.exp(k_squared_cache_cp[0] * tk) * u_data_k
            bar_f_hat_cp[0] = c_bar_1 / sin_k_x0_cp[0]
        else:
            j_indices = cp.arange(1, k)
            exp_terms = cp.exp(-k_squared_cache_cp[j_indices - 1] * tk)
            c_bar_prev = bar_f_hat_cp[j_indices - 1] * sin_k_x0_cp[j_indices - 1]
            sum_prev = cp.sum(exp_terms * c_bar_prev)

            n_k = _n_terms_forcing(k, float(tk))
            forcing_contrib = compute_forcing_contribution_at_time_gpu(
                F_gpu, float(tk), x0, n_terms=n_k, L=L
            )
            u_data_k = u_data_k - forcing_contrib

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


def compute_modified_t_j_nonhomogeneous_cupy(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    F: Callable[[float, float], float] | None = None,
    F_x: Callable[[float, float], float] | None = None,
    f_true: InitialCondition | None = None,
    L: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible entry point for CuPy implementation.

    Currently specialized to the default forcing F(x,t)=sin(x)*exp(-t) when
    F is not None and L=pi. For homogeneous (F=None), it uses only the
    homogeneous GPU path.
    """
    if F is None:
        # Purely homogeneous case: use the homogeneous GPU helper and finish
        # reconstruction on GPU.
        _, u_homogeneous_cp, _, _ = _compute_homogeneous_gpu(
            n=n, T=T, x0=x0, x_grid=x_grid, f_true=f_true, L=L
        )
        u_data_cp = u_homogeneous_cp
    else:
        if L != np.pi:
            msg = (
                "compute_modified_t_j_nonhomogeneous_cupy currently only supports "
                "F != None for L = pi."
            )
            raise ValueError(msg)

        # Use specialized default kernels for the CLI forcing.
        return compute_modified_t_j_nonhomogeneous_gpu(
            n=n,
            T=T,
            x0=x0,
            x_grid=x_grid,
            F_gpu=default_F_cupy,
            F_x_gpu=default_F_x_cupy,
            f_true=f_true,
            L=L,
        )

    # Homogeneous reconstruction only (F=None) using u_data_cp.
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

    t_vec_cp = cp.asarray(compute_t_vec(n, T))

    for k in range(1, max_k + 1):
        tk = t_vec_cp[k - 1]
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
    f_true_vals_np = np.asarray(
        (f_true or get_default_f_true)(x_grid), dtype=np.float64
    )
    bar_f_n_vals_np = cp.asnumpy(bar_f_n_vals_cp)

    return f_true_vals_np, bar_f_n_vals_np
