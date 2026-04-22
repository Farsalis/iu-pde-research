"""Core modules for PDE initial condition reconstruction.

Provides reconstruction algorithms (modified t_j homogeneous/non-homogeneous and
linear time sequence), forcing-term computations for the non-homogeneous heat
equation, and shared utilities: Fourier coefficients, default initial conditions,
and time vectors. Optional numba support for faster time-vector computation.
"""

from collections.abc import Callable

import numpy as np

from iu_pde_research.core.forcing import (
    compute_exp_decay_cutoff,
    compute_F_hat_j,
    compute_forcing_contribution_at_time,
    compute_forcing_truncation_constant_C,
    compute_n_k,
)
from iu_pde_research.core.forcing_autodiff import make_F_x_from_F
from iu_pde_research.core.reconstruction import (
    compute_linear_t_j_homogeneous,
    compute_modified_t_j,
    compute_modified_t_j_homogeneous_complex,
    compute_modified_t_j_nonhomogeneous,
)
from iu_pde_research.core.utils import (
    INITIAL_CONDITION_OPTIONS,
    NUMBA_AVAILABLE,
    get_default_f_true,
    get_f_true_sine_series_5,
    get_f_true_sine_series_10,
    get_f_true_sine_series_15,
)

try:
    from iu_pde_research.core.reconstruction_jax import compute_modified_t_j_jax
except ImportError:

    def compute_modified_t_j_jax(
        n: int,
        T: float,
        x0: float,
        x_grid: np.ndarray,
        f_true: Callable[[float | np.ndarray], float | np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback when JAX backend is unavailable."""
        msg = "JAX backend is unavailable. Install JAX to use compute_modified_t_j_jax."
        raise ImportError(msg)


try:
    from iu_pde_research.core.reconstruction_cupy import (
        compute_modified_t_j_nonhomogeneous_cupy,
    )
except ImportError:

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
        """Fallback when CuPy backend is unavailable."""
        msg = "CuPy backend is unavailable. Install CuPy to use GPU reconstruction."
        raise ImportError(msg)


# Export functions available in this package
__all__ = [
    "INITIAL_CONDITION_OPTIONS",
    "NUMBA_AVAILABLE",
    "compute_F_hat_j",
    "compute_exp_decay_cutoff",
    "compute_forcing_contribution_at_time",
    "compute_forcing_truncation_constant_C",
    "compute_linear_t_j_homogeneous",
    "compute_modified_t_j",
    "compute_modified_t_j_homogeneous_complex",
    "compute_modified_t_j_jax",
    "compute_modified_t_j_nonhomogeneous",
    "compute_modified_t_j_nonhomogeneous_cupy",
    "compute_n_k",
    "get_default_f_true",
    "get_f_true_sine_series_5",
    "get_f_true_sine_series_10",
    "get_f_true_sine_series_15",
    "make_F_x_from_F",
]
