"""Spectral basis helpers for heat-equation boundary conditions."""

from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

BoundaryCondition: TypeAlias = Literal["dirichlet", "neumann"]
ZeroModeMethod: TypeAlias = Literal["known", "measurement"]
ArrayLikeFloat: TypeAlias = float | NDArray[np.float64]


def validate_boundary_condition(bc: str) -> BoundaryCondition:
    """Validate and normalize a supported boundary condition name."""
    if bc in {"dirichlet", "neumann"}:
        return cast(BoundaryCondition, bc)

    msg = f"Unsupported boundary condition {bc!r}; expected 'dirichlet' or 'neumann'."
    raise ValueError(msg)


def validate_zero_mode_method(method: str) -> ZeroModeMethod:
    """Validate and normalize a supported Neumann zero-mode method name."""
    if method in {"known", "measurement"}:
        return cast(ZeroModeMethod, method)

    msg = f"Unsupported zero-mode method {method!r}; expected 'known' or 'measurement'."
    raise ValueError(msg)


def eigenvalue(j: int, L: float = np.pi, bc: BoundaryCondition = "dirichlet") -> float:
    """Return the Laplacian eigenvalue for mode ``j`` on ``[0, L]``."""
    validate_boundary_condition(bc)
    if L <= 0:
        msg = "Domain length L must be positive."
        raise ValueError(msg)
    if bc == "dirichlet" and j < 1:
        msg = "Dirichlet modes are indexed by positive integers."
        raise ValueError(msg)
    if bc == "neumann" and j < 0:
        msg = "Neumann modes are indexed by non-negative integers."
        raise ValueError(msg)

    if abs(L - np.pi) < 1e-10:
        return float(j**2)
    return float((j * np.pi / L) ** 2)


def mode_value(
    j: int,
    x: ArrayLikeFloat,
    L: float = np.pi,
    bc: BoundaryCondition = "dirichlet",
) -> ArrayLikeFloat:
    """Evaluate the boundary-condition eigenfunction at ``x``."""
    validate_boundary_condition(bc)
    if bc == "dirichlet":
        if j < 1:
            msg = "Dirichlet modes are indexed by positive integers."
            raise ValueError(msg)
        if abs(L - np.pi) < 1e-10:
            return cast(ArrayLikeFloat, np.sin(j * x))
        return cast(ArrayLikeFloat, np.sin(j * np.pi * x / L))

    if j < 0:
        msg = "Neumann modes are indexed by non-negative integers."
        raise ValueError(msg)
    if j == 0:
        if np.ndim(x) == 0:
            return 1.0
        return cast(ArrayLikeFloat, np.ones_like(x, dtype=np.float64))
    if abs(L - np.pi) < 1e-10:
        return cast(ArrayLikeFloat, np.cos(j * x))
    return cast(ArrayLikeFloat, np.cos(j * np.pi * x / L))


def coefficient_scale(
    j: int,
    L: float = np.pi,
    bc: BoundaryCondition = "dirichlet",
) -> float:
    """Return the Fourier coefficient normalization for mode ``j``."""
    validate_boundary_condition(bc)
    if L <= 0:
        msg = "Domain length L must be positive."
        raise ValueError(msg)
    if bc == "neumann" and j == 0:
        return float(1.0 / L)
    if j < 1:
        msg = f"{bc.capitalize()} nonzero modes are indexed by positive integers."
        raise ValueError(msg)
    return float(2.0 / L)
