"""JAX-accelerated reconstruction algorithms for the 1D heat equation.

This module provides GPU-friendly variants of the core reconstruction
algorithms, keeping a NumPy-like interface while using JAX under the hood.

Current coverage:
* compute_modified_t_j_jax: homogeneous modified t_j method.

The public API mirrors the CPU version in ``reconstruction.py`` as closely
as possible so callers can switch between CPU and GPU backends.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from iu_pde_research.core.utils import (
    InitialCondition,
    compute_t_vec,
    get_default_f_true,
)

# Use float64 so results match the SciPy/NumPy CPU path.
_jax_config_update = cast(Callable[[str, bool], None], jax.config.update)
_jax_config_update("jax_enable_x64", True)


def _jax_default_f_true(x: jnp.ndarray) -> jnp.ndarray:
    """Default initial condition expressed with JAX ops."""
    return (
        (1.0 / 8.0) * jnp.sin(2.0 * x)
        + (1.0 / 27.0) * jnp.sin(3.0 * x)
        + (1.0 / 64.0) * jnp.sin(4.0 * x)
        + (1.0 / 125.0) * jnp.sin(5.0 * x)
        + (1.0 / 216.0) * jnp.sin(6.0 * x)
    )


def _jax_vectorized_f_hat_default(
    j_vals: jnp.ndarray,
    *,
    L: float = np.pi,
    num_x: int = 512,
) -> jnp.ndarray:
    """Vectorized Fourier coefficients for the default f_true on a fixed grid.

    This replaces many small scipy.integrate.quad calls with a batched,
    GPU-friendly quadrature over a fixed spatial grid.
    """
    x_grid = jnp.linspace(0.0, float(L), num_x)
    f_vals = _jax_default_f_true(x_grid)

    if abs(L - np.pi) < 1e-10:
        sin_matrix = jnp.sin(j_vals[:, None] * x_grid[None, :])
    else:
        sin_matrix = jnp.sin(j_vals[:, None] * jnp.pi * x_grid[None, :] / float(L))

    integrand = f_vals[None, :] * sin_matrix
    # Trapezoidal rule along x; keep everything on device.
    integrals = jnp.trapezoid(integrand, x_grid, axis=1)
    return (2.0 / float(L)) * integrals


@jax.jit  # type: ignore[misc, unused-ignore]
def _compute_modified_t_j_kernel(
    t_vec: jnp.ndarray,
    x0: float,
    x_grid: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JAX kernel implementing the homogeneous modified t_j reconstruction.

    Args:
        t_vec: Time vector of shape (n,).
        x0: Spatial measurement location.
        x_grid: Spatial reconstruction grid, shape (m,).

    Returns:
        (f_true_vals, bar_f_n_vals) as JAX arrays on the current device.
    """
    n = t_vec.shape[0]

    # Synthetic measurements from sensor (homogeneous case).
    n_terms = 20
    j_vals = jnp.arange(1, n_terms + 1)
    f_hat_vals = _jax_vectorized_f_hat_default(j_vals)

    j_squared = j_vals**2
    sin_j_x0 = jnp.sin(j_vals * x0)

    # exp_matrix: (n, n_terms)
    exp_matrix = jnp.exp(-jnp.outer(t_vec, j_squared))
    u_data = jnp.sum(exp_matrix * f_hat_vals * sin_j_x0, axis=1)

    # Reconstruction of Fourier coefficients bar_f_hat.
    max_k = (n + 1) // 2  # ceil(n/2) with integer arithmetic
    k_indices = jnp.arange(1, max_k + 1)
    sin_k_x0 = jnp.sin(k_indices * x0)
    k_squared_cache = k_indices**2

    def body_fun(i: int, bar_f_hat: jnp.ndarray) -> jnp.ndarray:
        """Scan body over k index (0-based)."""
        tk = t_vec[i]

        def first_case(_: None) -> jnp.ndarray:
            value = jnp.exp(tk) * u_data[0] / sin_k_x0[0]
            return bar_f_hat.at[0].set(value)

        def general_case(_: None) -> jnp.ndarray:
            # j indices 1..i (1-based) -> 0..i-1 (0-based)
            # Use masking over the full vectors to keep shapes static.
            idx = jnp.arange(max_k)
            mask = idx < i

            exp_terms_full = jnp.exp(-k_squared_cache * tk)
            contrib_full = exp_terms_full * bar_f_hat * sin_k_x0
            sum_prev = jnp.sum(jnp.where(mask, contrib_full, 0.0))

            k_sq_tk = k_squared_cache[i] * tk
            value = jnp.exp(k_sq_tk) * (u_data[i] - sum_prev) / sin_k_x0[i]
            return bar_f_hat.at[i].set(value)

        return cast(
            jnp.ndarray,
            jax.lax.cond(i == 0, first_case, general_case, operand=None),
        )

    bar_f_hat0 = jnp.zeros(max_k, dtype=t_vec.dtype)
    bar_f_hat = jax.lax.fori_loop(0, max_k, body_fun, bar_f_hat0)

    k_vals = jnp.arange(1, max_k + 1)
    sin_matrix = jnp.sin(jnp.outer(x_grid, k_vals))
    bar_f_n_vals = sin_matrix @ bar_f_hat

    # True initial condition on x_grid (default homogeneous case).
    f_true_vals = _jax_default_f_true(x_grid)

    return f_true_vals, cast(jnp.ndarray, bar_f_n_vals)


def compute_modified_t_j_jax(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    f_true: InitialCondition | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """JAX-accelerated modified t_j reconstruction (homogeneous case).

    Args:
        n: Number of time measurements.
        T: Time scaling parameter.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction (NumPy array).
        f_true: Optional true initial condition. If None, uses the default
            initial condition. When provided, it is evaluated on CPU and only
            used to compute reference values; the reconstruction itself stays
            on the GPU.

    Returns:
        Tuple of NumPy arrays (f_true_vals, bar_f_n_vals) on x_grid.
    """
    if f_true is None:
        f_true = get_default_f_true

    # CPU-side time vector (NumPy) reused by both backends.
    t_vec_np = compute_t_vec(n, T)
    x_grid_np = np.asarray(x_grid, dtype=np.float64)

    # Move core arrays to device for the JAX kernel.
    t_vec = jnp.asarray(t_vec_np)
    x_grid_jax = jnp.asarray(x_grid_np)

    _, bar_f_n_vals_jax = _compute_modified_t_j_kernel(t_vec, float(x0), x_grid_jax)

    # Reference true values via user-supplied CPU function for consistency.
    f_true_vals_np = np.asarray(f_true(x_grid_np), dtype=np.float64)

    return f_true_vals_np, np.asarray(bar_f_n_vals_jax)
