"""Core modules for PDE initial condition reconstruction.

Provides reconstruction algorithms (modified t_j homogeneous/non-homogeneous and
linear time sequence), forcing-term computations for the non-homogeneous heat
equation, and shared utilities: Fourier coefficients, default initial conditions,
and time vectors. Optional numba support for faster time-vector computation.
"""

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
    compute_modified_t_j_nonhomogeneous,
)
from iu_pde_research.core.reconstruction_cupy import (
    compute_modified_t_j_nonhomogeneous_cupy,
)
from iu_pde_research.core.reconstruction_jax import compute_modified_t_j_jax
from iu_pde_research.core.utils import NUMBA_AVAILABLE

# Export functions available in this package
__all__ = [
    "NUMBA_AVAILABLE",
    "compute_F_hat_j",
    "compute_exp_decay_cutoff",
    "compute_forcing_contribution_at_time",
    "compute_forcing_truncation_constant_C",
    "compute_linear_t_j_homogeneous",
    "compute_modified_t_j",
    "compute_modified_t_j_nonhomogeneous",
    "compute_modified_t_j_nonhomogeneous_cupy",
    "compute_modified_t_j_jax",
    "compute_n_k",
    "make_F_x_from_F",
]
