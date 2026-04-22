"""Initial temperature profile reconstruction algorithms for PDE.

Entry points: compute_modified_t_j (homogeneous), compute_modified_t_j_nonhomogeneous
(with optional forcing F(x,t)), and compute_linear_t_j_homogeneous (linear time
sequence). Each returns (f_true_vals, bar_f_n_vals) on the given x_grid.

NOTE: We represent functions using numpy arrays (NDArray) for vectorized ops.
"""

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

from iu_pde_research.core.forcing import (
    compute_forcing_contribution_at_time,
    compute_forcing_truncation_constant_C,
    compute_n_k,
)
from iu_pde_research.core.forcing_autodiff import make_F_x_from_F
from iu_pde_research.core.utils import (
    compute_t_vec,
    get_default_f_true,
    get_f_hat_true,
)

# Optional cap on forcing modes n_k for numerical feasibility.
# Set to an integer (e.g. 200) to limit the number of forcing modes;
# leave as None to use the theoretical n_k from the paper without capping.
MAX_FORCING_MODES: int | None = None


def _f_hat_j_complex(
    j: int,
    f_true: Callable[[float], complex | float],
    L: float,
) -> np.complex128:
    """Fourier sine coefficient: (2/L) times integral of f(x) sin(j pi x / L).

    Supports complex-valued f.
    """

    def integrand_real(x: float) -> float:
        fv = f_true(x)
        z = (fv if isinstance(fv, complex) else complex(fv, 0.0)) * np.sin(
            j * np.pi * x / L
        )
        return float(z.real)

    def integrand_imag(x: float) -> float:
        fv = f_true(x)
        z = (fv if isinstance(fv, complex) else complex(fv, 0.0)) * np.sin(
            j * np.pi * x / L
        )
        return float(z.imag)

    re_h, _ = quad(integrand_real, 0.0, L)
    im_h, _ = quad(integrand_imag, 0.0, L)
    return np.complex128((2.0 / L) * complex(re_h, im_h))


def compute_modified_t_j_homogeneous_complex(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    f_true: Callable[[float], complex | float] | None = None,
    L: float = np.pi,
    n_terms: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a complex initial condition using the modified t_j method (homogeneous).

    Uses the same stabilizing time sequence as the real-valued routine and the
    sine-basis forward substitution from the complex Dirichlet heat formulation.

    Args:
        n: Number of time measurements.
        T: Time scaling parameter for the modified factorial sequence.
        x0: Spatial measurement point in (0, L).
        x_grid: Spatial grid for evaluating true and reconstructed profiles.
        f_true: Initial condition mapping [0, L] to complex numbers (real f
            allowed). If None, uses the package default extended with zero
            imaginary part.
        L: Domain length. Defaults to π.
        n_terms: Number of sine modes in the forward synthetic sensor model.

    Returns:
        Tuple ``(f_true_vals, bar_f_n_vals)`` of complex arrays on ``x_grid``
        giving the true initial data and the reconstructed truncated sine series.
    """
    if f_true is None:

        def f_true(x: float) -> complex:
            return complex(get_default_f_true(x), 0.0)

    t_vec = compute_t_vec(n, T)

    j_vals = np.arange(1, n_terms + 1)
    f_hat_vals = np.array(
        [_f_hat_j_complex(int(j), f_true, L) for j in j_vals],
        dtype=np.complex128,
    )

    if abs(L - np.pi) < 1e-10:
        j_squared = j_vals.astype(np.float64) ** 2
        sin_j_x0 = np.sin(j_vals * x0)
    else:
        j_squared = np.power(j_vals * np.pi / L, 2).astype(np.float64)
        sin_j_x0 = np.sin(j_vals * np.pi * x0 / L)

    exp_matrix = np.exp(-np.outer(t_vec, j_squared))
    u_data = np.sum(exp_matrix * f_hat_vals * sin_j_x0, axis=1)

    n_bar = int(np.ceil(n / 2))
    bar_f_hat = np.zeros(n_bar, dtype=np.complex128)
    k_range = np.arange(1, n_bar + 1)

    if abs(L - np.pi) < 1e-10:
        sin_k_x0 = np.sin(k_range * x0)
        k_squared_cache = k_range.astype(np.float64) ** 2
    else:
        sin_k_x0 = np.sin(k_range * np.pi * x0 / L)
        k_squared_cache = np.power(k_range * np.pi / L, 2).astype(np.float64)

    for k in range(1, n_bar + 1):
        tk = t_vec[k - 1]
        if k == 1:
            bar_f_hat[0] = np.exp(k_squared_cache[0] * tk) * u_data[0] / sin_k_x0[0]
        else:
            j_indices = np.arange(1, k)
            exp_terms = np.exp(-k_squared_cache[j_indices - 1] * tk)
            sum_prev = np.sum(
                exp_terms * bar_f_hat[j_indices - 1] * sin_k_x0[j_indices - 1]
            )
            bar_f_hat[k - 1] = (
                np.exp(k_squared_cache[k - 1] * tk)
                * (u_data[k - 1] - sum_prev)
                / sin_k_x0[k - 1]
            )

    if abs(L - np.pi) < 1e-10:
        sin_matrix = np.sin(np.outer(x_grid, k_range))
    else:
        sin_matrix = np.sin(np.outer(x_grid, k_range * np.pi / L))

    bar_f_n_vals = sin_matrix @ bar_f_hat

    f_true_vals = np.array(
        [f_true(float(x)) for x in x_grid],
        dtype=np.complex128,
    )

    return f_true_vals, bar_f_n_vals


def compute_modified_t_j(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    f_true: Callable[[float], float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct initial condition using the modified t_j method (homogeneous).

    Same layout as the homogeneous approximation scheme.

    Args:
        n: Number of time measurements.
        T: Time scaling parameter.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction.
        f_true: True initial condition function. If None, uses the default
            initial condition.

    Returns:
        Tuple of `(f_true_vals, bar_f_n_vals)` where `f_true_vals` is the true
        initial condition on `x_grid` and `bar_f_n_vals` is the reconstructed
        initial condition on `x_grid`.
    """
    if f_true is None:
        f_true = get_default_f_true

    t_vec = compute_t_vec(n, T)  # Compute t_k as a vector and then optimize

    n_terms = 20  # Number of measurements from sensor
    j_vals = np.arange(1, n_terms + 1)
    f_hat_vals = np.array(
        [get_f_hat_true(int(j), f_true) for j in j_vals]
    )  # Generating j true Fourier coefficients
    j_squared = j_vals**2
    sin_j_x0 = np.sin(j_vals * x0)
    exp_matrix = np.exp(
        -np.outer(t_vec, j_squared)
    )  # Computing dot product since these are arrays
    u_data = np.sum(exp_matrix * f_hat_vals * sin_j_x0, axis=1)

    bar_f_hat = np.zeros(int(np.ceil(n / 2)))
    sin_k_x0 = np.sin(np.arange(1, len(bar_f_hat) + 1) * x0)
    max_k = len(bar_f_hat)
    j_squared_cache = np.arange(1, max_k + 1) ** 2

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
    f_true_vals = np.asarray(
        [float(f_true(float(x))) for x in x_grid], dtype=np.float64
    )

    return f_true_vals, bar_f_n_vals  # True measurements and reconstructed measurements


def compute_modified_t_j_nonhomogeneous(
    n: int,
    T: float,
    x0: float,
    x_grid: np.ndarray,
    F: Callable[[float, float], float] | None = None,
    F_x: Callable[[float, float], float] | None = None,
    f_true: Callable[[float], float] | None = None,
    L: float = np.pi,
    *,
    backend: str = "cpu",
    F_gpu: Callable[..., object] | None = None,
    F_x_gpu: Callable[..., object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct initial condition using the modified t_j method (non-homogeneous).

    Same layout as the non-homogeneous approximation scheme. When F is provided,
    either supply F_x (spatial derivative of F) or ensure F is JAX-differentiable
    (uses jax.numpy) so that F_x is computed automatically for the n_k truncation.

    Args:
        n: Number of time measurements.
        T: Time scaling parameter.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction.
        F: Forcing function F(x, t). If None, assumes the homogeneous case.
        F_x: Spatial derivative F_x(x, s) = ∂F/∂x. If None and F is not None,
            F_x is obtained via JAX autodiff from F (F must use JAX ops).
        f_true: True initial condition function. If None, uses the default
            initial condition.
        L: Domain length. Defaults to pi.

    Returns:
        Tuple of `(f_true_vals, bar_f_n_vals)` where `f_true_vals` is the true
        initial condition on `x_grid` and `bar_f_n_vals` is the reconstructed
        initial condition on `x_grid`.
    """
    if backend not in {"cpu", "gpu"}:
        msg = f"Unsupported backend {backend!r} (expected 'cpu' or 'gpu')."
        raise ValueError(msg)

    # GPU backend is implemented in reconstruction_cupy; we import lazily to
    # avoid importing CuPy when not needed.
    if backend == "gpu":
        try:
            from iu_pde_research.core.reconstruction_cupy import (
                compute_modified_t_j_nonhomogeneous_gpu,
            )
        except ImportError as exc:  # pragma: no cover - hardware/env dependent
            msg = (
                "backend='gpu' requested but CuPy-based implementation is not "
                "available or CuPy is not installed."
            )
            raise RuntimeError(msg) from exc

        if F is None:
            msg = (
                "backend='gpu' for compute_modified_t_j_nonhomogeneous requires "
                "a non-None forcing F."
            )
            raise ValueError(msg)

        if F_gpu is None or F_x_gpu is None:
            msg = (
                "backend='gpu' requires CuPy-compatible F_gpu and F_x_gpu "
                "callables; provide them explicitly or use backend='cpu'."
            )
            raise ValueError(msg)

        return compute_modified_t_j_nonhomogeneous_gpu(
            n=n,
            T=T,
            x0=x0,
            x_grid=x_grid,
            F_gpu=F_gpu,
            F_x_gpu=F_x_gpu,
            f_true=f_true,
            L=L,
        )

    # --- CPU backend (default) ----------------------------------------------
    t_vec = compute_t_vec(n, T)

    if f_true is None:
        f_true = get_default_f_true

    if F is not None and F_x is None:
        F_x = make_F_x_from_F(F)

    C: float = 0.0
    if F is not None and F_x is not None:
        C = compute_forcing_truncation_constant_C(F_x, T, L)

    n_terms = 50
    j_vals = np.arange(1, n_terms + 1)

    # Synthetic measurements representing those from sensor for (x0, t_k)
    if abs(L - np.pi) < 1e-10:  # Equal to pi
        f_hat_vals = np.array(
            [
                quad(lambda x, j_val=j: f_true(x) * np.sin(j_val * x), 0, L)[0] * 2 / L
                for j in j_vals
            ]
        )

        j_squared = j_vals**2
        sin_j_x0 = np.sin(j_vals * x0)
    else:
        f_hat_vals = np.array(
            [
                quad(
                    lambda x, j_val=j: f_true(x) * np.sin(j_val * np.pi * x / L),
                    0,
                    L,
                )[0]
                * 2
                / L
                for j in j_vals
            ]
        )

        j_squared = np.power(j_vals * np.pi / L, 2).astype(np.float64)
        sin_j_x0 = np.sin(j_vals * np.pi * x0 / L)

    # Compute homogeneous part of NH case
    exp_matrix = np.exp(-np.outer(t_vec, j_squared))
    u_homogeneous = np.sum(exp_matrix * f_hat_vals * sin_j_x0, axis=1)

    # Compute forcing contribution to u(x0, t_k) with truncation n_k per step.
    if F is not None:
        u_forcing = np.array(
            [
                compute_forcing_contribution_at_time(
                    F,
                    tk,
                    x0,
                    (
                        min(compute_n_k(k, tk, C), MAX_FORCING_MODES)
                        if MAX_FORCING_MODES is not None
                        else compute_n_k(k, tk, C)
                    ),
                    L,
                )
                for k, tk in enumerate(t_vec, start=1)
            ]
        )
        u_data = u_homogeneous + u_forcing
    else:
        u_data = u_homogeneous

    # Apply homogeneous reconstruction algorithm, getting rid of forcing contributions
    bar_f_hat = np.zeros(int(np.ceil(n / 2)))
    max_k = len(bar_f_hat)

    if abs(L - np.pi) < 1e-10:
        sin_k_x0 = np.sin(np.arange(1, max_k + 1) * x0)
        k_squared_cache = np.arange(1, max_k + 1) ** 2
    else:
        sin_k_x0 = np.sin(np.arange(1, max_k + 1) * np.pi * x0 / L)
        k_squared_cache = np.power(np.arange(1, max_k + 1) * np.pi / L, 2).astype(
            np.float64
        )

    for k in range(1, max_k + 1):
        tk = t_vec[k - 1]
        u_data_k = u_data[k - 1]

        if k == 1:
            if F is not None:
                n_k = compute_n_k(k, tk, C)
                if MAX_FORCING_MODES is not None:
                    n_k = min(n_k, MAX_FORCING_MODES)
                forcing_contrib = compute_forcing_contribution_at_time(
                    F, tk, x0, n_k, L
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
                n_k = compute_n_k(k, tk, C)
                if MAX_FORCING_MODES is not None:
                    n_k = min(n_k, MAX_FORCING_MODES)
                forcing_contrib = compute_forcing_contribution_at_time(
                    F, tk, x0, n_k, L
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
    f_true_vals = np.asarray(
        [float(f_true(float(x))) for x in x_grid], dtype=np.float64
    )

    return f_true_vals, bar_f_n_vals


def compute_linear_t_j_homogeneous(
    n: int,
    t0: float,
    x0: float,
    x_grid: np.ndarray,
    f_true: Callable[[float], float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct initial condition using a linear time sequence method.

    Args:
        n: Number of time measurements.
        t0: Initial time step.
        x0: Spatial measurement point.
        x_grid: Grid of x values for reconstruction.
        f_true: True initial condition function. If None, uses the default
            initial condition.

    Returns:
        Tuple of `(f_true_vals, bar_f_n_vals)` where `f_true_vals` is the true
        initial condition on `x_grid` and `bar_f_n_vals` is the reconstructed
        initial condition on `x_grid`.
    """
    if f_true is None:
        f_true = get_default_f_true

    def t_k(k: int, t_param: float) -> float:
        """Compute time for measurement k."""
        return (n + k - 1) * t_param

    t_vec = np.array([t_k(k, t0) for k in range(1, n + 1)])

    def f_hat_true(j: int) -> float:
        """Compute Fourier coefficient for mode j."""

        def integrand(x: float) -> float:
            return float(f_true(x) * np.sin(j * x))

        result, _ = quad(integrand, 0, np.pi)
        return float(result * 2 / np.pi)

    def u_xtk(x0_val: float, t_vec_in: np.ndarray, n_terms: int = 50) -> np.ndarray:
        """Compute solution at x0 for given times."""
        results: list[float] = []

        for t in t_vec_in:
            sum_val = 0.0
            for j in range(1, n_terms + 1):
                sum_val += np.exp(-(j**2) * t) * f_hat_true(j) * np.sin(j * x0_val)
            results.append(sum_val)

        return np.array(results)

    u_data = u_xtk(x0, t_vec)
    bar_f_hat = np.zeros(int(np.ceil(n / 2)))

    for k in range(1, len(bar_f_hat) + 1):
        tk = t_vec[k - 1]

        if k == 1:
            bar_f_hat[0] = np.exp(tk) * u_data[0] / np.sin(x0)
        else:
            sum_prev = sum(
                [
                    np.exp(-(j**2) * tk) * bar_f_hat[j - 1] * np.sin(j * x0)
                    for j in range(1, k)
                ]
            )

            bar_f_hat[k - 1] = (
                np.exp(k**2 * tk) * (u_data[k - 1] - sum_prev) / np.sin(k * x0)
            )

    def bar_f_n(x: float) -> float:
        """Reconstructed function."""
        return float(
            sum(bar_f_hat[k - 1] * np.sin(k * x) for k in range(1, len(bar_f_hat) + 1))
        )

    f_true_vals = f_true(x_grid)
    bar_f_n_vals = np.array([bar_f_n(float(x)) for x in x_grid])

    return f_true_vals, bar_f_n_vals
