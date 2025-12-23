"""Initial temperature profile reconstruction algorithms for PDE.

NOTE: We represent functions using numpy arrays NDArray[any] in order to use numpy's vectorized operations.
"""

import numpy as np
from scipy.integrate import quad
from typing import Callable, Optional, Tuple

from core.utils import (
    compute_t_vec,
    get_default_f_true,
    get_f_hat_true,
)
from core.forcing import compute_forcing_contribution_at_time


def compute_modified_t_j(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    f_true: Optional[Callable[[float], float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct initial condition using modified t_j method (homogeneous). Same layout as homogeneous approximation scheme.

    Args:
        n: Number of time measurements.
        T: Time scaling parameter.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction.
        f_true: True initial condition function. If None, uses default.

    Returns:
        Tuple of (f_true_vals, bar_f_n_vals) where:
            f_true_vals: True initial condition on x_grid.
            bar_f_n_vals: Reconstructed initial condition on x_grid.
    """
    if f_true is None:
        f_true = get_default_f_true

    t_vec = compute_t_vec(n, T)  # Compute t_k as a vector and then optimize

    n_terms = 20  # Number of measurements from sensor
    j_vals = np.arange(1, n_terms + 1)
    f_hat_vals = np.array([get_f_hat_true(j, f_true) for j in j_vals])  # Generating j true Fourier coefficients
    j_squared = j_vals**2
    sin_j_x0 = np.sin(j_vals * x0)
    exp_matrix = np.exp(-np.outer(t_vec, j_squared))  # Computing dot product since these are arrays
    u_data = np.sum(exp_matrix * f_hat_vals * sin_j_x0, axis=1)

    bar_f_hat = np.zeros(int(np.ceil(n / 2)))
    sin_k_x0 = np.sin(np.arange(1, len(bar_f_hat) + 1) * x0)
    max_k = len(bar_f_hat)
    j_squared_cache = np.arange(1, max_k + 1)**2

    for k in range(1, max_k + 1):  # Amount of terms we reconstruct based on bar_f_hat
        tk = t_vec[k - 1]
        if k == 1:
            bar_f_hat[0] = np.exp(tk) * u_data[0] / sin_k_x0[0]
        else:
            j_indices = np.arange(1, k)
            exp_terms = np.exp(-j_squared_cache[j_indices - 1] * tk)
            sum_prev = np.sum(
                exp_terms * bar_f_hat[j_indices - 1] * sin_j_x0[j_indices - 1]
            )
            k_squared_tk = j_squared_cache[k - 1] * tk
            bar_f_hat[k - 1] = (
                np.exp(k_squared_tk) * (u_data[k - 1] - sum_prev) / sin_k_x0[k - 1]
            )

    k_vals = np.arange(1, len(bar_f_hat) + 1)
    sin_matrix = np.sin(np.outer(x_grid, k_vals))
    bar_f_n_vals = sin_matrix @ bar_f_hat
    f_true_vals = f_true(x_grid)

    return f_true_vals, bar_f_n_vals  # True measurements and reconstructed measurements


def compute_modified_t_j_nonhomogeneous(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    F: Optional[Callable[[float, float], float]] = None,
    f_true: Optional[Callable[[float], float]] = None,
    L: float = np.pi,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct initial condition using modified t_j method (non-homogeneous). Same layout as non-homogeneous approximation scheme.

    Args:
        n: Number of time measurements.
        T: Time scaling parameter.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction.
        F: Forcing function F(x, t). If None, assumes homogeneous case.
        f_true: True initial condition function. If None, uses default.
        L: Domain length. Defaults to pi.

    Returns:
        Tuple of (f_true_vals, bar_f_n_vals) where:
            f_true_vals: True initial condition on x_grid.
            bar_f_n_vals: Reconstructed initial condition on x_grid.
    """
    t_vec = compute_t_vec(n, T)

    if f_true is None:
        f_true = get_default_f_true

    n_terms = 20
    j_vals = np.arange(1, n_terms + 1)

    # Synthetic measurements representing those from sensor for (x0, t_k)
    if abs(L - np.pi) < 1e-10:  # Equal to pi
        f_hat_vals = np.array([
            quad(lambda x: f_true(x) * np.sin(j * x), 0, L)[0] * 2 / L
            for j in j_vals
        ])

        j_squared = j_vals**2
        sin_j_x0 = np.sin(j_vals * x0)
    else:
        f_hat_vals = np.array([
            quad(lambda x: f_true(x) * np.sin(j * np.pi * x / L), 0, L)[0] * 2 / L
            for j in j_vals
        ])

        j_squared = (j_vals * np.pi / L)**2
        sin_j_x0 = np.sin(j_vals * np.pi * x0 / L)

    # Compute homogeneous part of NH case
    exp_matrix = np.exp(-np.outer(t_vec, j_squared))
    u_homogeneous = np.sum(exp_matrix * f_hat_vals * sin_j_x0, axis=1)

    # Compute forcing contribution to u(x0, t_k)
    if F is not None:
        u_forcing = np.array([
            compute_forcing_contribution_at_time(F, tk, x0, n_terms, L)
            for tk in t_vec
        ])
        u_data = u_homogeneous + u_forcing
    else:
        u_data = u_homogeneous

    # Apply homogeneous reconstruction algorithm, getting rid of forcing contributions
    bar_f_hat = np.zeros(int(np.ceil(n / 2)))
    max_k = len(bar_f_hat)

    if abs(L - np.pi) < 1e-10:
        sin_k_x0 = np.sin(np.arange(1, max_k + 1) * x0)
        k_squared_cache = np.arange(1, max_k + 1)**2
    else:
        sin_k_x0 = np.sin(np.arange(1, max_k + 1) * np.pi * x0 / L)
        k_squared_cache = (np.arange(1, max_k + 1) * np.pi / L)**2

    for k in range(1, max_k + 1):
        tk = t_vec[k - 1]
        u_data_k = u_data[k - 1]

        if k == 1:
            if F is not None:
                forcing_contrib = compute_forcing_contribution_at_time(
                    F, tk, x0, n_terms, L
                )

                u_data_k = u_data_k - forcing_contrib

            c_bar_1 = np.exp(k_squared_cache[0] * tk) * u_data_k
            bar_f_hat[0] = c_bar_1 / sin_k_x0[0]
        else:
            j_indices = np.arange(1, k)
            exp_terms = np.exp(-k_squared_cache[j_indices - 1] * tk)

            c_bar_prev = bar_f_hat[j_indices - 1] * sin_k_x0[j_indices - 1]
            sum_prev = np.sum(exp_terms * c_bar_prev)

            if F is not None:
                forcing_contrib = compute_forcing_contribution_at_time(
                    F, tk, x0, n_terms, L
                )

                u_data_k = u_data_k - forcing_contrib

            c_bar_k = np.exp(k_squared_cache[k - 1] * tk) * (u_data_k - sum_prev)
            bar_f_hat[k - 1] = c_bar_k / sin_k_x0[k - 1]

    k_vals = np.arange(1, len(bar_f_hat) + 1)

    if abs(L - np.pi) < 1e-10:
        sin_matrix = np.sin(np.outer(x_grid, k_vals))
    else:
        sin_matrix = np.sin(np.outer(x_grid, k_vals * np.pi / L))

    bar_f_n_vals = sin_matrix @ bar_f_hat  # Take regular dot product of NDArrays
    f_true_vals = f_true(x_grid)

    return f_true_vals, bar_f_n_vals


def compute_linear_t_j_homogeneous(
    n: int,
    t0: float,
    x0: float,
    x_grid: np.ndarray,
    f_true: Optional[Callable[[float], float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct initial condition using linear time sequence method.

    Args:
        n: Number of time measurements.
        t0: Initial time step.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction.
        f_true: True initial condition function. If None, uses default.

    Returns:
        Tuple of (f_true_vals, bar_f_n_vals) where:
            f_true_vals: True initial condition on x_grid.
            bar_f_n_vals: Reconstructed initial condition on x_grid.
    """
    if f_true is None:
        f_true = get_default_f_true

    def t_k(k, T):
        """Compute time for measurement k."""
        return (n + k - 1) * T

    t_vec = np.array([t_k(k, t0) for k in range(1, n + 1)])

    def f_hat_true(j):
        """Compute Fourier coefficient for mode j."""
        integrand = lambda x: f_true(x) * np.sin(j * x)
        result, _ = quad(integrand, 0, np.pi)

        return result * 2 / np.pi

    def u_xtk(x0, t_k, n_terms=50):
        """Compute solution at x0 for given times."""
        results = []

        for t in t_k:
            sum_val = 0
            for j in range(1, n_terms + 1):
                sum_val += np.exp(-j**2 * t) * f_hat_true(j) * np.sin(j * x0)
            results.append(sum_val)

        return np.array(results)

    u_data = u_xtk(x0, t_vec)
    bar_f_hat = np.zeros(int(np.ceil(n / 2)))

    for k in range(1, len(bar_f_hat) + 1):
        tk = t_vec[k - 1]

        if k == 1:
            bar_f_hat[0] = np.exp(tk) * u_data[0] / np.sin(x0)
        else:
            sum_prev = sum([
                np.exp(-j**2 * tk) * bar_f_hat[j - 1] * np.sin(j * x0)
                for j in range(1, k)
            ])
            
            bar_f_hat[k - 1] = (
                np.exp(k**2 * tk) * (u_data[k - 1] - sum_prev) / np.sin(k * x0)
            )

    def bar_f_n(x):
        """Reconstructed function."""
        return sum([
            bar_f_hat[k - 1] * np.sin(k * x)
            for k in range(1, len(bar_f_hat) + 1)
        ])

    f_true_vals = f_true(x_grid)
    bar_f_n_vals = np.array([bar_f_n(x) for x in x_grid])

    return f_true_vals, bar_f_n_vals

